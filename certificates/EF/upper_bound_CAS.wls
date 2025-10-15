$Assumptions = {Element[_, Reals]};
$Assumptions = 
 Join[$Assumptions, {0 < \[Mu] < L, \[Mu] > 0, L > 0, L > \[Mu], 0 < \[Epsilon], \[Epsilon] <= 1}]

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