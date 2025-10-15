$Assumptions = {Element[_, Reals]};
$Assumptions = Join[$Assumptions, {0 < \[Mu] < L, \[Mu] > 0, L > 0, L > \[Mu], 0 < \[Epsilon], \[Epsilon] <= 1}];

(* Encodes the same rate as Theorem 1 *)
\[Eta] = 2/(L + \[Mu]) (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]);
\[Lambda] = \[Eta]/(L + \[Mu]) ((1 - Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);
\[Rho] = Sqrt[\[Epsilon]] + 1/4 (1 + Sqrt[\[Epsilon]]) (L - \[Mu]) \[Lambda];

EF21 = {xk1 -> xk - \[Eta] dk};
ineq1 = fk - fk1 + (gk1 - gk)^2/(2 L) + gk (xk1 - xk) + \[Mu]/(2 (1 - \[Mu]/L)) (xk - xk1 - 1/L (gk - gk1))^2;
ineq2 = fk1 - fk + (gk - gk1)^2/(2 L) + gk1 (xk - xk1) + \[Mu]/(2 (1 - \[Mu]/L)) (xk1 - xk - 1/L (gk1 - gk))^2;
ineqEC = (gk1 - dk)^2 - 2 (gk1 - dk) EC + EC2 - \[Epsilon] (gk1 - dk)^2;

(* This is the weighted sum mentioned at the beginning of the proof. *)
\[Lambda]' = Sqrt[\[Epsilon]]/((\[Eta] (L + \[Mu]))) ((1 - Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);
weightedSum1 = \[Lambda]' (ineq1 + ineq2) + ineqEC // FullSimplify;
weightedSum1 = weightedSum1 /. EF21 // FullSimplify;

(* This is what we claim it is equivalent to. If this is true, then the rest of the proof is easily verified by hand. *)
a = \[Rho] - \[Epsilon] + \[Lambda]' \[Eta]^2 L \[Mu]/(L - \[Mu]);
b = \[Eta]/2 \[Lambda]' (L + \[Mu])/(L - \[Mu]);

const = 1/(2 + Sqrt[\[Epsilon]]);
LyapunovEF21 = (gk - dk)^2 + Sqrt[\[Epsilon]] gk^2;
LyapunovEF21p = (gk1 - dk)^2 - 2 (gk1 - dk) EC + EC2 + Sqrt[\[Epsilon]] gk1^2;
weightedSum2 = LyapunovEF21p - \[Rho] LyapunovEF21 + a (dk + 1/a ((\[Epsilon] + b) gk1 - (\[Rho] + b) gk))^2 // Simplify;

(* The two are equivalent, and hence the rate is valid. *)
Print["Verifying the rate shown in Theorem 2..."];
If[TrueQ[Simplify[weightedSum1 == weightedSum2]], 
  Print["The rate shown in Theorem 2 is valid."], 
  Print["The rate shown in Theorem 2 is NOT valid."]];