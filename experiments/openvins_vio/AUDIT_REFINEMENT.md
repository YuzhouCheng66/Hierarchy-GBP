# Audit refinement before formal timing

The frozen production algorithm and trajectory/covariance/increment admission
rules are unchanged. An additional FP64 innovation-statistic equality assertion,
beyond the gate-decision requirement in PROTOCOL.md, exposed an unreliable dense
reference on ill-conditioned rejected features during the first full-sequence
audit. These failed runs are retained as audit01/audit02 evidence.

Example: V1_01, 21 clones, an 85-row projected feature has a maximum Jacobian
entry of 7.82e9 and innovation condition estimate 4.30e11. The candidate FP64
statistic is 105515549870.253; the directly formed dense reference gives
141436202137.994; the cutoff is 107.522. An independent 70-decimal-digit mpmath
calculation gives 105515565696.164 for the exact stored root model and
105515565696.133 for the independent stored dense covariance model. All reject.
The candidate's relative error is approximately 1.50e-7, while the two accurately
evaluated models agree to about 2.9e-13. Forming H P H^T in ordinary double
precision is a poor oracle for this feature.

Final audit behavior: if the two FP64 statistics differ beyond 1e-10 + 1e-6 times
the reference magnitude, evaluate both models independently in 113-bit binary
arithmetic. Require their statistics to agree within the same mixed tolerance,
and require the production accept/reject decision to match both. Record how many
features use this check, the maximum model discrepancy and production FP64 error.
Otherwise use the independent dense FP64 gate. No high-precision computation is
used when audit mode is disabled. No accepted/rejected label is changed by audit.

Covariance and tangent-increment checks retain their original tolerances. Full
sequences are rerun with the final audit executable before any formal timing.
This documents a correction to the numerical oracle, not deletion of difficult
sequences or an algorithm change introduced to improve measured speed.
