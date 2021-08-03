\begin{table}[]
\caption{}
\label{tab:my-table}
\begin{tabular}{c|c|c|c}
\hline
Methods & \begin{tabular}[c]{@{}c@{}}Same Bag \\ (\textit\{do(D)\})\end{tabular} & \begin{tabular}[c]{@{}c@{}}De-confound\\  (\textit\{do(X)\})\end{tabular} & Direct Effect \\ \hline
SimpleMIL \cite\{SimpleMIL\} & - & - & TE \\
Patch-based CNN \cite\{hou2016cvpr\} & \CheckmarkBold & - & NDE \\
RCEMIL \cite\{RCE\} & \CheckmarkBold & - & NDE \\
Top-\textit\{k\}+Center  \cite\{chikontwe2020topk\} & \CheckmarkBold & - & NDE \\ \hline
Our method & - & \CheckmarkBold & TE \\ \hline
\end{tabular}
\end{table}