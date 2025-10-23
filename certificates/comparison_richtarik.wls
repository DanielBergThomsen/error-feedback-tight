ClearAll["Global`*"];
$Assumptions = {Element[_, Reals], 0 < \[Mu], \[Mu] < L, 
   0 < \[Epsilon], \[Epsilon] < 1, \[Kappa] > 1};

Print["Checking proof that the article's rate is smaller than \
existing rates for EF/EF21..."]

(* Setup rate of EF/EF21 *)
\[Eta] = 2/(L + \[Mu]) (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]]);
\[Lambda] = \[Eta]/(L + \[Mu]) ((1 - 
        Sqrt[\[Epsilon]]) (L - \[Mu]) + (1 + 
        Sqrt[\[Epsilon]]) Sqrt[(L - \[Mu])^2 + 
        16 L \[Mu] Sqrt[\[Epsilon]]/(1 + Sqrt[\[Epsilon]])^2]);
\[Rho] = 
  Sqrt[\[Epsilon]] + 1/4 (1 + Sqrt[\[Epsilon]]) (L - \[Mu]) \[Lambda];
kRho = \[Rho] /. {\[Mu] -> L/\[Kappa]} // FullSimplify;

(* Set up the rate of Richtarik *)
\[Beta] = \[Epsilon]/(1 - Sqrt[\[Epsilon]]);
\[Theta] = 1 - Sqrt[\[Epsilon]];
richtarik = 1 - (1 - Sqrt[\[Epsilon]])/(1 + Sqrt[\[Epsilon]])/\[Kappa];

(* Quickly check that the rate is equal to the one given in the paper \
*)
diff = richtarik - kRho // FullSimplify;
If[FullSimplify[
  diff != (1 - 
     Sqrt[\[Epsilon]] - (1 - 
        Sqrt[\[Epsilon]])/((1 + Sqrt[\[Epsilon]]) \[Kappa]) - ((1 - 
          Sqrt[\[Epsilon]]) (\[Kappa] - 1)^2 (1 - Sqrt[\[Epsilon]] + 
          Sqrt[1 + \[Epsilon] + (2 Sqrt[\[Epsilon]] (1 + \[Kappa] (6 \
+ \[Kappa])))/(\[Kappa] - 1)^2]))/(2 (\[Kappa] + 1)^2))], 
 Print["WARNING: error in proof"]]

(* These two are equivalent *)
\[CapitalDelta] = -((-1 + 
        Sqrt[\[Epsilon]]) (-2 + \[Kappa] (-3 + \[Epsilon] (-1 + \
\[Kappa])^2 + 
           2 Sqrt[\[Epsilon]] (1 + \[Kappa])^2 + (1 + Sqrt[\[Epsilon]])*
            R + \[Kappa] (4 + \[Kappa] - (1 + Sqrt[\[Epsilon]])*
               R)))) // FullSimplify;
Rexpr = Sqrt[(-1 + \[Kappa])^2 + \[Epsilon] (-1 + \[Kappa])^2 + 
   2 Sqrt[\[Epsilon]] (1 + \[Kappa] (6 + \[Kappa]))];
If[FullSimplify[\[CapitalDelta] != 
     FullSimplify[Numerator[Together[diff]]] /. {R -> Rexpr}], 
  Print["WARNING: error in proof"]];

(* Check that other terms outbalance R times its coefficient *)
coefR = Coefficient[\[CapitalDelta], R] // FullSimplify;
\[CapitalDelta]SansR = \[CapitalDelta] - coefR * R // FullSimplify;
positiveFactor = (1 + \[Kappa] (1 + \[Kappa] (-5 + 
          3 \[Kappa]) + \[Epsilon] (1 + \[Kappa]) (-1 + 3 \[Kappa]) + 
       2 Sqrt[\[Epsilon]] (-1 + \[Kappa]) (1 + 3 \[Kappa])));
If[FullSimplify[\[CapitalDelta]SansR^2 - (coefR*Rexpr)^2 != 
    4*(1 - Sqrt[\[Epsilon]])^2 (1 + \[Kappa])^2*positiveFactor], 
  Print["WARNING: error in proof"]];

(* positiveFactor is the only factor in the RHS that has not yet been v\
erified to be positive. *)
If[FullSimplify[Coefficient[positiveFactor, Sqrt[\[Epsilon]]] <= 0], 
  Print["WARNING: error in proof"]];
If[FullSimplify[Coefficient[positiveFactor, \[Epsilon]] <= 0], 
  Print["WARNING: error in proof"]];

Print["If we made it here without warnings, we are good!"]