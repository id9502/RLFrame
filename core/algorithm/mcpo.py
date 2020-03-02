import torch
from torch.optim.lbfgs import LBFGS
from core.math import mmd, cg, safe_div
from torch.distributions.normal import Normal
from core.model.policy_with_value import PolicyWithValue
from core.common import StepDictList, SampleBatch, ParamDict


# global cache for speed-up mmd calculation
mmd_cache = {}


def mcpo_optim(g, b, c, Hvp_f, delta, bc_step_f):
    # min  f(x)           => g @ (x - x0),
    # s.t. g1(x) <= d     => b @ (x - x0) <= d - g1(x0) = c
    #      g2(x) <= delta => 1/2 * (x - x0) @ H @ (x - x0) <= delta

    # dx = -1/lambda * H^-1 @ (g + nu*b)
    # nu = max(0, -(lambda*c + r)/s)
    # lambda:
    #    nu = 0:  fa(lambda) = -1/2*(q/lambda + 2*delta*lambda)
    #               lambda_a = sqrt(0.5*q/delta)
    #    nu != 0: fb(lambda) = -1/2*((q - r^2/s)/lambda + (2*delta - c^2/s)*lambda) + r*c/s
    #               lambda_b = sqrt((q - r^2/s)/(2*delta - c^2/s))

    # q = g @ H^-1 @ g >= 0
    # r = g @ H^-1 @ b
    # s = b @ H^-1 @ b >= 0

    IHbp = cg(Hvp_f, b, nsteps=20)
    IHgp = cg(Hvp_f, g, nsteps=20)
    q = g @ IHgp
    r = g @ IHbp
    s = b @ IHbp

    # -- check intersection -- #
    # with min_x x @ H @ x; s.t. b @ x = c
    # optimal x = c/s * H^-1 @ b
    # if x @ H @ x = c^2 / s > delta and mmd constraint meets,
    #       none of mmd constraints active, pure TRPO can be used
    # if x @ H @ x = c^2 / s > delta but mmd constraint breaks,
    #       current problem is infeasible, pure MMD-BC can be used
    # if x @ H @ x = c^2 / s < delta,
    #       there should be at least one x that satisfies constraints, use dual solver to find solution
    if safe_div(c*c, s) > 2. * delta:
        if c >= 0:
            # pure TRPO here, no limit check needs in line search
            # dx     = -1/lambda * H^-1 @ g
            # lambda = sqrt(0.5*q/delta)
            lam = (0.5 * q / delta) ** 0.5
            dx = - safe_div(IHgp, lam)
            line_search_check_range = (-1., -1.)
            print("inside TRPO branch, c={}".format(c))
            return dx, line_search_check_range, 1
        else:
            # pure MMD-BC here, no limit check needs in line search
            x = bc_step_f()
            line_search_check_range = None
            print("inside BC branch, c={}".format(c))

            return x, line_search_check_range, 4
    else:
        # dual solver here, line search should be done when x is outside min-circle
        # -- get lambda -- #
        lam_mid = -safe_div(r, c)
        L_mid = -0.5 * (safe_div(q, lam_mid) + 2. * delta * lam_mid)

        # lambda_a: nu = 0; constraint deactivate
        lam_a = (0.5 * q / delta) ** 0.5
        L_a = -(q * delta) ** 0.5

        # lambda_b: nu != 0; constraint activate
        lam_b = safe_div(q * s - r * r, 2. * delta * s - c * c) ** 0.5
        L_b = safe_div(-safe_div(q * s - r * r, 2. * delta * s - c * c) ** 0.5 + r * c, s)

        if lam_mid > 0.:
            if c >= 0.:
                # current feasible
                # lam_a in [0, lam_mid)
                # lam_b in (lam_mid, infty)
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    L_a = L_mid
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    L_b = L_mid
            else:
                # current infeasible
                # lam_a in (lam_mid, infty)
                # lam_b in [0, lam_mid)
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    L_a = L_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    L_b = L_mid
            if L_a >= L_b:
                lam = lam_a
            else:
                lam = lam_b
        else:
            if c >= 0.:
                lam = lam_b
            else:
                lam = lam_a

        # -- get nu -- #
        nu = torch.clamp_min(-safe_div(lam * c + r, s), 0.)

        # -- get dx -- #
        dx = -safe_div(IHgp + nu * IHbp, lam)

        # -- get check range -- #
        step = b @ dx
        if c >= 0.:
            # current feasible
            if step >= c:
                # step larger than min feasible range
                line_search_check_range = (safe_div(c, step), 1.)
            else:
                # step smaller than min feasible range
                line_search_check_range = (-1., -1.)
        else:
            # current infeasible
            if step <= c:
                # step larger than min feasible range
                line_search_check_range = (0., safe_div(c, step))
            else:
                # step smaller than min feasible range
                # strictly, this should not occur, but actually it will happen due to numerical issue
                line_search_check_range = (0., 1.)
        # print("inside MC-PO branch, feasible={}, lam={}, nu={}".format(bool(c >= 0), lam, nu))
        return dx, line_search_check_range, (2 if c >= 0. else 3)


def line_search(policy_net, expect_df, f0, f, g1, d, fullstep, linesearch_check_range, steps=10, accept_ratio=0.1):
    # min  f(x),
    # s.t. g1(x) <= d
    #      g2(x) <= delta
    x0 = policy_net.get_flat_params().detach()
    with torch.no_grad():
        for stepfrac in [.5 ** x for x in range(steps)]:
            fullstep.clamp_(-1.e5, 1.e5)
            x_new = x0 + stepfrac * fullstep
            policy_net.set_flat_params(x_new)
            if linesearch_check_range[0] < stepfrac < linesearch_check_range[1]:
                g1_new = g1()
                if g1_new > d:
                    continue
            f_new = f()
            actual_improve = f_new - f0
            expected_improve = expect_df * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio or (-1.e-7 < actual_improve < 1.e-7 and -1.e-7 < f0 < 1.e-7):
                return True, x_new
        return False, x0


def get_tensor(batch, demo, device):
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    demo_states = demo[0] if demo is not None else None
    demo_actions = demo[1] if demo is not None else None
    return states, actions, advantages, returns, demo_states, demo_actions


def update_value_net(value_net, states, returns, l2_reg):
    optimizer = LBFGS(value_net.parameters(), max_iter=25, history_size=5)

    def closure():
        optimizer.zero_grad()
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss = value_loss + param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss

    optimizer.step(closure)


def mcpo_step(config: ParamDict, batch: SampleBatch, policy: PolicyWithValue, demo: StepDictList = None):
    global mmd_cache
    max_kl, damping, l2_reg, bc_method, d_init, d_factor, d_max, i_iter = \
        config.require("max kl", "damping", "l2 reg", "bc method", "constraint", "constraint factor", "constraint max",
                       "current training iter")
    states, actions, advantages, returns, demo_states, demo_actions = \
        get_tensor(batch, demo, policy.device)

    # ---- annealing on constraint tolerance ---- #
    d = min(d_init + d_factor * i_iter, d_max)

    # ---- update critic ---- #
    update_value_net(policy.value_net, states, returns, l2_reg)

    # ---- update policy ---- #
    with torch.no_grad():
        fixed_log_probs = policy.policy_net.get_log_prob(states, actions).detach()

    # ---- define the reward loss function for MCPO ---- #
    def RL_loss():
        log_probs = policy.policy_net.get_log_prob(states, actions)
        action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
        return action_loss.mean()

    # ---- define the reward loss function for MCPO ---- #
    def Dkl():
        kl = policy.policy_net.get_kl(states)
        return kl.mean()

    # ---- define MMD constraint from demonstrations ---- #
    # ---- if use oversample, recommended value is 5 ---- #
    def Dmmd(oversample=0):
        from_demo = torch.cat((demo_states, demo_actions), dim=1)
        # here uses distance between (s_e, a_e) and (s_e, pi_fix(s_e)) instead of (s, a) if not exact

        a, _, var = policy.policy_net(demo_states)
        if oversample > 0:
            sample_a = Normal(torch.zeros_like(a), torch.ones_like(a)).sample((oversample,)) * var + a
            sample_s = demo_states.expand(oversample, -1, -1)
            from_policy = torch.cat((sample_s, sample_a), dim=2).view(-1, s.size(-1) + a.size(-1))
        else:
            from_policy = torch.cat((demo_states, a + 0. * var), dim=1)

        return mmd(from_policy, from_demo, mmd_cache)

    def Dl2():
        import torch.nn.functional as F
        # here we use the mean
        mean, logstd = policy.policy_net(demo_states)
        return F.mse_loss(mean + 0. * logstd, demo_actions)

    # ---- define BC from demonstrations ---- #
    if bc_method == "l2":
        Dc = Dl2
    else:
        Dc = Dmmd

    def DO_BC():
        assert demo_states is not None, "I should not arrive here with demos == None"
        dist = Dc()
        if dist > d:
            policy_optimizer = torch.optim.Adam(policy.policy_net.parameters())
            #print(f"Debug: Constraint not meet, refining tile it satisfies {dist} < {d}")
            for _ in range(500):
                policy_optimizer.zero_grad()
                dist.backward()
                policy_optimizer.step()
                dist = Dc()
                if dist < d:
                    break
            #print(f"Debug: BC margin={d - dist}")
        else:
            print(f"Debug: constraint meet, {dist.item()} < {d}")
        x = policy.policy_net.get_flat_params().detach()
        return x

    # ---- define grad funcs ---- #
    def Hvp_f(v, damping=damping):
        kl = Dkl()

        grads = torch.autograd.grad(kl, policy.policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v.detach()).sum()
        grads = torch.autograd.grad(kl_v, policy.policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    f0 = RL_loss()
    grads = torch.autograd.grad(f0, policy.policy_net.parameters())
    g = torch.cat([grad.view(-1) for grad in grads]).detach()
    f0.detach_()

    if demo_states is None:
        d_value = -1.e7
        c = 1.e7
        b = torch.zeros_like(g)
    else:
        d_value = Dc()
        c = (d - d_value).detach()
        grads = torch.autograd.grad(d_value, policy.policy_net.parameters())
        b = torch.cat([grad.view(-1) for grad in grads]).detach()

    # ---- update policy net with CG-LineSearch algorithm---- #
    d_theta, line_search_check_range, case = mcpo_optim(g, b, c, Hvp_f, max_kl, DO_BC)

    if torch.isnan(d_theta).any():
        if torch.isnan(b).any():
            print("b is NaN when Dc={}. Rejecting this step!".format(d_value))
        else:
            print("net parameter is NaN. Rejecting this step!")
        success = False
    elif line_search_check_range is not None:
        expected_df = g @ d_theta
        success, new_params = line_search(policy.policy_net, expected_df, f0, RL_loss, Dc, d, d_theta, line_search_check_range)
        policy.policy_net.set_flat_params(new_params)
    else:
        # here d_theta is from BC, so skip line search procedure
        success = True
    return (case if success else -1), d_value
