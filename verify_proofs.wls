(* verify_proofs.wls *)
$Assumptions = {Element[_, Reals]};
$Assumptions = 
 Join[$Assumptions, {0 < \[Mu] < L, \[Mu] > 0, L > 0, L > \[Mu], 0 < \[Epsilon], \[Epsilon] <= 1}]

(* -------------------------------------- We begin verifying the proof of the rate in Theorem 1. -------------------------------------- *)
\[Eta] = 2 / (L + \[Mu]) (1 - Sqrt[\[Epsilon]]) / (1 + Sqrt[\[Epsilon]]);
\[Lambda] = \[Eta] / (L + \[Mu]) ((1 - Sqrt[\[Epsilon]])(L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]] / (1 + Sqrt[\[Epsilon]])^2]);
\[Rho] = Sqrt[\[Epsilon]] + 1 / 4 (1 + Sqrt[\[Epsilon]]) (L - \[Mu]) \[Lambda];
\[Nu] = 1 / Sqrt[\[Epsilon]];

(* Encode interpolation conditions *)
ineq1 = fk - fs - gk (xk - xs) + 1 / (2L) (gk)^2 + \[Mu] / (2 (1 - \[Mu] / L)) (xk - xs - 1 / L gk)^2;
ineq2 = fs - fk + 1 / (2L) gk^2 + \[Mu] / (2 (1 - \[Mu] / L)) (xk - xs - 1 / L gk)^2;
ineqC = (ek + \[Eta] gk - ck)^2 - \[Epsilon] (ek + \[Eta] gk)^2;

(* This is the weighted sum mentioned at the beginning of the proof. *)
weightedSum1 = \[Lambda] (ineq1 + ineq2) + \[Nu] ineqC;

(* This is what we claim it is equivalent to. If this is true, then the rest of the proof is easily verified by hand. *)
a = (\[Rho] - Sqrt[\[Epsilon]]) * (1 + Sqrt[\[Epsilon]]) / Sqrt[\[Epsilon]];
const = 1 / (2 + 1 / Sqrt[\[Epsilon]])
LyapunovEF[xk_, ek_] := (const * (xk - xs)^2 - 2 * const * (xk - xs) * ek + (1 - const) * ek^2) / const
weightedSum2 = LyapunovEF[xk - ck, ek + \[Eta] gk - ck] - \[Rho] LyapunovEF[xk, ek] + a (ek - (\[Rho] - 1) / a (xk - xs) + 2(Sqrt[\[Epsilon]] - 1) / (a(L + \[Mu])) gk)^2;

(* The two are equivalent, and hence the rate is valid. *)
Print["Verifying the rate shown in Theorem 1..."];
If[
  TrueQ[FullSimplify[weightedSum1 == weightedSum2]],
  Print["The rate shown in Theorem 1 is valid."],
  Print["The rate shown in Theorem 1 is NOT valid."]
];

(* -------------------------------------- We now verify the proof of the rate in Theorem 2. -------------------------------------- *)
(* Since it is the same rate, uses the same optimal step size and the same Lagrange multiplier, we only need to redefine interpolation conditions in our setup. *)
EF21 = {xk1 -> xk - \[Eta] dk, dk1 -> dk + ck};
ineq1 = fk - fk1 + (gk1 - gk)^2 / (2L) + gk (xk1 - xk) + \[Mu] / (2(1 - \[Mu] / L)) (xk - xk1 - 1 / L (gk - gk1))^2;
ineq2 = fk1 - fk + (gk - gk1)^2 / (2L) + gk1 (xk - xk1) + \[Mu] / (2(1 - \[Mu] / L)) (xk1 - xk - 1 / L (gk1 - gk))^2;
ineqC = (gk1 - dk - ck)^2 - \[Epsilon] (gk1 - dk)^2;

(* This is the weighted sum mentioned at the beginning of the proof. *)
\[Lambda]' = Sqrt[\[Epsilon]] / ((\[Eta] (L + \[Mu]))) ((1 - Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]] / (1 + Sqrt[\[Epsilon]])^2]);
weightedSum1 = \[Lambda]' (ineq1 + ineq2) + ineqC // FullSimplify;
weightedSum1 = weightedSum1 /. EF21 // FullSimplify;

(* This is what we claim it is equivalent to. If this is true, then the rest of the proof is easily verified by hand. *)
a = \[Rho] - \[Epsilon] + \[Lambda]' \[Eta]^2 L \[Mu] / (L - \[Mu]);
b = \[Eta] / 2 \[Lambda]' (L + \[Mu]) / (L - \[Mu]);

const = 1 / (2 + Sqrt[\[Epsilon]])
LyapunovEF21[gk_, dk_] := ((1-const) * gk^2 - 2 * const * gk * dk + const * dk^2) / const
weightedSum2 = LyapunovEF21[gk1, dk1] - \[Rho] LyapunovEF21[gk, dk] + a (dk + 1 / a ((\[Epsilon] + b) gk1 - (\[Rho] + b) gk))^2 // Simplify;
weightedSum2 = weightedSum2 /. EF21 // Simplify;

(* The two are equivalent, and hence the rate is valid. *)
Print["Verifying the rate shown in Theorem 2..."];
If[
  TrueQ[Simplify[weightedSum1 == weightedSum2]],
  Print["The rate shown in Theorem 2 is valid."],
  Print["The rate shown in Theorem 2 is NOT valid."]
];