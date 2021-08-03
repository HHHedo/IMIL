\begin{table}[]
\begin{tabular}{lccccc}
\hline
\multicolumn{1}{l|}{\multirow{2}{*}{parameters}} & \multicolumn{5}{c}{Digestpath} \\ \cline{2-6} 
\multicolumn{1}{l|}{} & \multicolumn{1}{l}{AUC} & \multicolumn{1}{l}{Acc} & \multicolumn{1}{l}{F1} & \multicolumn{1}{l}{Recall} & \multicolumn{1}{l}{Precision} \\ \hline
\multicolumn{6}{l}{(a) Momentum $m$} \\ \hline
\multicolumn{1}{l|}{$m$ = 0} & 92.18 & 86.69 & 73.55 & 79.25 & 68.80 \\
\multicolumn{1}{l|}{$m$ = 0.25} & 91.97 & 86.72 & 73.37 & 78.16 & 69.25 \\
\multicolumn{1}{l|}{$m$ = 0.5} & 92.06 & 86.84 & 73.74 & 78.77 & 69.60 \\
\multicolumn{1}{l|}{$m$ = 0.75} & 92.47 & 87.32 & 74.68 & 79.32 & 70.70 \\
\multicolumn{1}{l|}{$m$ = 0.9} & 91.60 & 85.56 & 72.88 & 79.75 & 68.07 \\ \hline
\multicolumn{6}{l}{(b) Threshold $T$} \\ \hline
\multicolumn{1}{l|}{$T $ = 0.9} & 92.17 & 87.35 & 74.39 & 77.89 & 71.62 \\
\multicolumn{1}{l|}{$T $ = 0.95} & 92.06 & 86.84 & 73.74 & 78.77 & 69.60 \\
\multicolumn{1}{l|}{$T $ = 1} & 92.23 & 87.35 & 74.34 & 77.94 & 71.46 \\
\multicolumn{1}{l|}{$T $ = 1.05}  & 91.28 & 86.79 & 72.23 & 74.12 & 70.49 \\ \hline
\multicolumn{6}{l}{(c) Step $\tau$} \\ \hline
\multicolumn{1}{l|}{$\tau$ = 0.025} & 91.99 & 86.90 & 73.88 & 78.02 & 70.46 \\
\multicolumn{1}{l|}{$\tau$ = 0.05} & 92.06 & 86.84 & 73.74 & 78.77 & 69.60 \\
\multicolumn{1}{l|}{$\tau$ = 0.075} & 92.46 & 87.43 & 74.56 & 78.20 & 71.46 \\
\multicolumn{1}{l|}{$\tau$ = 0.01} & 91.59 & 87.06 & 73.40 & 77.29 & 69.93 \\ \hline
\end{tabular}
\end{table}