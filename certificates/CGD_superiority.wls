ClearAll["Global`*"];
$Assumptions = {Element[_, Reals], 0 < \[Mu], \[Mu] < L, 
  0 < \[Epsilon], \[Epsilon] < 1, \[Kappa] > 1};

(* Optimal step size and multiplier for EF *)
\[Eta] = 2/(L + \[Mu]) (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]);
\[Lambda] = \[Eta]/(L + \[Mu]) ((1 - 
        Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + 
        Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 
        16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);

(* Convergence rate, and conversion to \kappa *)
\[Rho]EF = 
  Sqrt[\[Epsilon]] + 1/4 (1 + Sqrt[\[Epsilon]]) (L - \[Mu]) \[Lambda] /. {\[Mu] -> L/\[Kappa]} // FullSimplify;

(* CGD optimal rate *)
keps = L /\[Mu] (1 + Sqrt[\[Epsilon]])/(1 - Sqrt[\[Epsilon]]);
\[Rho]CGD = ((keps - 1)/(keps + 1))^2 //. {L / \[Mu] -> \[Kappa]};

Print["Verifying CGD rate < EF rate..."];
If[
  TrueQ[Assuming[0 < \[Epsilon] < 1 && \[Kappa] > 1, Reduce[FullSimplify[\[Rho]EF - \[Rho]CGD > 0]]]],
  Print["CGD is superior."],
  Print["Something went wrong!"]
];