import numpy as np   
import tensorflow as tf 


def policy_val(t, yf, eff_pred):
    """ 
    Compute the value of policy risk.

    Parameters:
    - t: np.array, treatment assignment (binary: 0 or 1)
    - yf: np.array, observed outcomes
    - eff_pred: np.array, predicted treatment effects

    Returns:
    - policy_value: float, computed value of policy risk
    """

    t, yf, eff_pred = map(np.asarray, (t, yf, eff_pred))

    if np.any(np.isnan(eff_pred)):
        return np.nan

    policy = (eff_pred > 0) + 0.0
    treat_overlap = (policy == t) * (t > 0)
    control_overlap = (policy == t) * (t < 1)

    if np.sum(treat_overlap) == 0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap) == 0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit * treat_value + (1 - pit) * control_value

    return policy_value


def evaluate_att_policy(
        model, 
        input_tensor, 
        test_idx, 
        true_att, 
        true_y, 
        rct_flags
    ):
    """ 
    Evaluate ATT (Average Treatment Effect on the Treated) 
    and policy risk using an estimated model.

    Parameters:
    - model: Trained model used for predictions.
    - input_tensor: tf.Tensor, input data including treatment assignments.
    - test_idx: list or array, indices for test set.
    - true_att: float, ground truth ATT value.
    - true_y: array, observed outcomes.
    - rct_flags: bool, indicating whether evulate use a subset of RCT.

    Returns:
    - pol_risk: float, policy risk (1 - policy value).
    - err_att: float, absolute error in ATT estimation.
    """

    pre_T, pre_C = model.pre_no_interf(input_tensor, test_idx, only_t=True)

    all_t = input_tensor[:, -1]
    t_test = tf.gather(all_t, test_idx)
    y_test = tf.gather(true_y, test_idx)

    only_t_ITE = pre_T - pre_C
    ATT = tf.reduce_mean(only_t_ITE)

    pre_1, pre_0 = model.pre_no_interf(input_tensor, test_idx, only_t=False)
    ITE = pre_1 - pre_0

    test_rct_flags = np.array(rct_flags)[test_idx]
    rct_idx = tf.where(test_rct_flags > 0)[:, 0]

    t_rct = np.array(tf.gather(t_test, rct_idx)).reshape(-1)
    y_rct = np.array(tf.gather(y_test, rct_idx)).reshape(-1)
    rct_ITE = np.array(tf.gather(ITE, rct_idx)).reshape(-1)

    pol = policy_val(t_rct, y_rct, rct_ITE)
    pol_risk = 1 - pol

    err_att =np.abs(ATT - true_att)

    return pol_risk, err_att


def evaluate_ate_pehe(
        model, 
        input_tensor, 
        test_idx, 
        true_ite, 
        true_ate
    ):
    """Evaluate ATE and PEHE error metrics.

    Args:
        model (tf.keras.Model): Trained causal inference model.
        input_tensor (np.ndarray or tf.Tensor): Input data for the test set.
        test_idx (list or np.ndarray): Indices of the test set samples.
        true_ite (np.ndarray): Ground-truth Individual Treatment Effects (ITE).
        true_ate (float): Ground-truth Average Treatment Effect (ATE).

    Returns:
        tuple: (PEHE error, ATE error)
    """

    pre_y1, pre_y0 = model(input_tensor, test_idx)

    pred_ite = pre_y1 - pre_y0
    pred_ate = tf.reduce_mean(pred_ite)

    pehe = np.mean((pred_ite - true_ite) ** 2)
    err_ate = np.abs(pred_ate - true_ate)

    return pehe, err_ate


def evalate(
        model, 
        input_tensor, 
        test_idx, 
        true_ate, 
        true_y, 
        true_ite=[], 
        rct_flags=None
    ):
    """Evaluate causal inference performance using either PEHE/polich risk and ATT.

    Args:
        model (tf.keras.Model): Trained causal inference model.
        input_tensor (np.ndarray or tf.Tensor): Input data for the test set.
        test_idx (list or np.ndarray): Indices of the test set samples.
        true_ate (float): Ground-truth Average Treatment Effect (ATE).
        true_y (np.ndarray): Observed outcomes for the test set.
        true_ite (np.ndarray, optional): Ground-truth Individual Treatment Effects (ITE). Defaults to None.
        rct_flags (np.ndarray, optional): Flags indicating randomized controlled trial (RCT) samples. Defaults to None.

    Returns:
        tuple: (PEHE error, ATE error) if true ITE is provided, else (Policy Risk, ATT error).
    """

    if len(true_ite) > 0:
        pehe, err_ate = evaluate_ate_pehe(model, input_tensor, test_idx, true_ite, true_ate)
        return pehe, err_ate

    pol_risk, err_att = evaluate_att_policy(model, input_tensor, test_idx, true_ate, true_y, rct_flags)

    return pol_risk, err_att

