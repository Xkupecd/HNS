\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{FEIstyle}[2025/02/05 v. 2.0.0 Thesis template - FEI STUBA]

% parse arguments

\newcommand*\myset{a4paper,12pt}
\newcommand*\parentclass{article}

\LoadClass[\myset]{\parentclass}

\usepackage[overload]{textcase}

% hyperref has to be loaded *before* glossaries
% see https://mirror.kumi.systems/ctan/macros/latex/contrib/glossaries/glossaries-user.html#sec:hyperref
\usepackage[bookmarksopen,colorlinks,plainpages=false,urlcolor=blue,hypertexnames=false,unicode]{hyperref}

\usepackage[acronym,nopostdot,nogroupskip,nonumberlist]{glossaries}

\usepackage[final, babel, nopatch=item]{microtype}
\usepackage[utf8]{inputenc}
% we should use ISO 690 & 690-2 http://www.fei.stuba.sk/sk/kniznica-fei/vzory-bibliografickych-odkazov-a-citovanie.html?page_id=1756 
% package for biblatex https://github.com/michal-h21/biblatex-iso690
\usepackage[backend=biber,
            sorting=none,
            style=iso-numeric,
            shortnumeration=true,
            citestyle=numeric-comp,
            maxnames=5,
            minnames=1,
            giveninits=true]{biblatex}
\DefineBibliographyStrings{slovak}{
  urlfrom = {Dostupné z}
}

\usepackage[main=slovak,english]{babel}
\languageattribute{slovak}{split}

\usepackage[style=slovak]{csquotes}
\usepackage[T1]{fontenc}

\usepackage[immediate]{silence}
\WarningFilter[temp]{latex}{Command}
% silence warnings of sectsty/sectsty.sty:
%     'Command \underbar has changed.'
%     'Command \underline has changed.'

\usepackage{sectsty}
\DeactivateWarningFilters[temp]

\usepackage{chngcntr}
\usepackage{tocloft}
\usepackage{rotating}
\usepackage[a4paper, tmargin=3cm,bmargin=3cm,lmargin=2.75cm,rmargin=2.75cm]{geometry}
\usepackage{xifthen}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{upgreek}
\usepackage{isomath}
\usepackage{ifmtarg}
\usepackage{xstring}
\usepackage[compact]{titlesec}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{lmodern}
\usepackage{changepage}
\usepackage{scrextend}
\usepackage{xurl}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{pdfpages}
\usepackage[all]{nowidow}
%\usepackage{showframe} % debugging overflows

% \cftsetpnumwidth{1.8em} % set the dots width in loX (f,t,l,a...) https://tex.stackexchange.com/q/369283/106389

% https://stackoverflow.com/a/74266238
\lstset{
  extendedchars=true,
  literate=
    {á}{{\'a}}1
    {ä}{{\"a}}1
    {č}{{\v{c}}}1
    {ď}{{\v{d}}}1
    {é}{{\'e}}1
    {í}{{\'i}}1
    {ĺ}{{\'l}}1
    {ľ}{{\v{l}}}1
    {ň}{{\v{n}}}1
    {ó}{{\'o}}1
    {ô}{{\^o}}1
    {ŕ}{{\'{r}}}1
    {š}{{\v{s}}}1
    {ť}{{\v{t}}}1
    {ú}{{\'u}}1
    {ý}{{\'y}}1
    {ž}{{\v{z}}}1
    {Á}{{\'A}}1
    {Ä}{{\""A}}1
    {Č}{{\v{C}}}1
    {Ď}{{\v{C}}}1
    {É}{{\'E}}1
    {Í}{{\'I}}1
    {Ĺ}{{\'L}}1
    {Ľ}{{\v{L}}}1
    {Ň}{{\v{N}}}1
    {Ó}{{\'O}}1
    {Ô}{{\^O}}1
    {Ŕ}{{\'{R}}}1
    {Š}{{\v{S}}}1
    {Ť}{{\v{T}}}1
    {Ú}{{\'U}}1
    {Ý}{{\'Y}}1
    {Ž}{{\v{Z}}}1
}

% https://www.overleaf.com/learn/latex/Code_listing
\definecolor{codegreen}{rgb}{0,0.506,0}
\definecolor{codegray}{rgb}{0.2,0.2,0.2}
\definecolor{codeviolet}{rgb}{0.58,0,0.82}
\definecolor{codemagenta}{rgb}{0.839,0,0.494}
\definecolor{backcolour}{rgb}{0.95,0.95,0.95}
\lstdefinestyle{code-listing}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{codemagenta},
    numberstyle=\ttfamily\footnotesize\color{codegray},
    stringstyle=\color{codeviolet},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=none,
    numbersep=8pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    frame=single,
    framesep=3pt,
    framerule=0pt,
    xleftmargin=4pt,
    xrightmargin=4pt,
    framexleftmargin=1pt,
    framexrightmargin=1pt
}



% deine global variables 
\def\@nedefinovane{! UNDEFINED: \textbackslash}

\def\@skola{Slovensk\a'a technick\a'a  univerzita v Bratislave}
\def\FEIschool#1{\def\@skola{#1}}
\newcommand{\printSkola}{\@skola}

\def\@fakulta{Fakulta elektrotechniky a informatiky}
\def\FEIfaculty#1{\def\@fakulta{#1}}
\newcommand{\printFakulta}{\@fakulta}

\def\@autor{\@nedefinovane FEIauthor}
\def\FEIauthor#1{\def\@autor{#1}}
\newcommand{\printAutor}{\@autor}

\def\@nazov{\@nedefinovane FEItitle}
\def\FEItitle#1{\def\@nazov{#1}}
\newcommand{\printNazov}{\@nazov}

\def\@predmet{\@nedefinovane FEIpredmet}
\def\FEIpredmet#1{\def\@predmet{#1}}
\newcommand{\printPredmet}{\@predmet}


\def\@rok{\number\year}
\newcommand\FEIdate[1]{
    \def\@rok{#1}}




%definovanie globalnych nastaveni
\pagenumbering{arabic}  
\pagestyle{plain}
\parindent=0.63cm
\parskip=1pt
\linespread{1.25}
\hypersetup{colorlinks, citecolor=black, filecolor=black, linkcolor=black, urlcolor=black}
\sectionfont{\fontsize{20pt}{18pt}\selectfont}
\subsectionfont{\fontsize{16pt}{15pt}\selectfont}
\subsubsectionfont{\fontsize{14pt}{14pt}\selectfont}

\titlespacing*\section{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*\subsection{0pt}{3ex plus 1ex minus .2ex}{1.25ex plus .2ex}
\titlespacing*\subsubsection{0pt}{2.5ex plus 1ex minus .2ex}{1.25ex plus .2ex}

\lstset{
  basicstyle=\scriptsize\ttfamily,
  columns=flexible, % removes spaces
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false, % removes ␣ character from " "
  captionpos=b % caption below text
}


% http://tex.stackexchange.com/questions/588/how-can-i-change-the-margins-for-only-part-of-the-text
\def\changemargin#1#2{\list{}{\rightmargin#2\leftmargin#1}\item[]}
\let\endchangemargin=\endlist 

\newcommand{\dir}[3]{
\begin{description}[topsep=0pt,leftmargin=0.5cm,style=nextline]
  \item[#1] $\cdot$ #2
\end{description}
\ifx&#3&%
   % #3 is empty
\else
    \vspace{-5mm}
    \begin{changemargin}{0.5cm}{0cm}
        #3
    \end{changemargin}
    \vspace{-4mm}
\fi
}

\newcommand{\file}[2]{
    \begin{description}[topsep=0pt,leftmargin=0.5cm,font=\normalfont\itshape,style=nextline]
        \item[#1] $\cdot$ #2
    \end{description}
}

\newcommand\FEIcover{
    \pagestyle{empty}
    \newgeometry{tmargin=2cm,bmargin=1.8cm,lmargin=2.75cm,rmargin=2.75cm}
                \def\@skolaTemp{\printSkola}
                \def\@skolaFakulta{\printFakulta}
                \def\@skolaNazovPrace{\printNazov}
		   \def\@skolaPredmet{\printPredmet}
        \begin{sffamily}
        \begin{center}
            {\fontsize{14.5pt}{0}\selectfont\bfseries \MakeTextUppercase{\@skolaTemp}\\[11.3pt] \@skolaFakulta}
        \end{center}
        \vspace{11.5mm}
	  \vspace{51.5mm}
        \begin{center}
            {\fontsize{20.5pt}{18.2pt}\selectfont\bfseries\nointerlineskip\@skolaNazovPrace\par}\vspace{29pt}
            {\fontsize{14.5pt}{0}\selectfont\bfseries \@skolaPredmet}
        \end{center}
        \vspace{\fill}
        \begin{flushleft}
            {\fontsize{14.5pt}{0}\selectfont\bfseries
                \@rok
                \hfill
                \printAutor}
        \end{flushleft} 
    \end{sffamily}
    \restoregeometry
    \newpage    
}
       
\newcommand\FEIcontent{ 
    \pagestyle{plain}
    \addtocontents{toc}{\protect\thispagestyle{plain}}
    \tableofcontents
    \newpage
}
        
\newcommand\listoffiguresandtables{
    \renewcommand{\cftfigpresnum}{Obrázok\ }
    \renewcommand{\cfttabpresnum}{Tabu\v{l}ka\ } 
    \newlength{\mylenf}
    \settowidth{\mylenf}{\cftfigpresnum}
    \setlength{\cftfignumwidth}{\dimexpr\mylenf+3em}
    \setlength{\cfttabnumwidth}{\dimexpr\mylenf+3em}
    \section*{\fontsize{20pt}{1.3}\selectfont Zoznam obr\'{a}zkov a tabuliek}
      
    \phantomsection
    \@starttoc{lof}
    \bigskip
    \@starttoc{lot}
}
        
\newcommand\FEIlistOfTables{
    \pagestyle{plain}        
    \listoftables
    \newpage
    \pagestyle{plain}
}
        
\newcommand\FEIlistOfFigures{   
    \pagestyle{plain}
    \listoffigures
    \newpage
    \pagestyle{plain}
}
        
\newcommand\FEIlistOfAlgorithms{
    \def\@zoznamAlg{Zoznam algoritmov}
    \renewcommand{\listalgorithmname}{\fontsize{20pt}{1.3}\selectfont \@zoznamAlg}
    \pagestyle{plain}
    \listofalgorithms
    \newpage
    \pagestyle{plain}
}

        
\newcommand\FEIlistOfFiguresAndTables{ 
    \pagestyle{plain}
    \listoffiguresandtables
    \newpage
    \pagestyle{plain}
}

\renewcommand{\glsnamefont}[1]{\textbf{#1}}

% https://tex.stackexchange.com/questions/266591/printing-glossaries-with-long-styles-adds-to-the-tables-counter
\newglossarystyle{FEIlong}{
    \setglossarystyle{long}
    \renewenvironment{theglossary}{
      \begin{longtable*}{lp{\glsdescwidth}}
    }
    {\end{longtable*}}%
}

\newcommand\FEIglossaries[1]{
    \setglossarystyle{FEIlong}
    \makeglossaries
    \loadglsentries{#1}
    \setlength\LTleft{8pt}
}

\newcommand\FEIlistOfGlossaries{
    	\def\@zoznamZn{Zoznam značiek a skratiek}
    %\addtocontents{lof}{\protect\thispagestyle{plain}}
    %\addtocontents{lot}{\protect\thispagestyle{plain}}
    \printglossary[type=\acronymtype,title={\fontsize{20pt}{1.3}\selectfont \@zoznamZn}]
    \newpage
}

\newcommand\FEImanualListOfGlossaries[1]{
       \def\@zoznamZn{Zoznam značiek a skratiek}
    \section*{\@zoznamZn}
    {
        \linespread{1.7}\selectfont
        \input{#1}\par
    }
    \newpage
}

\newcommand\FEIintroduction[1]{ 
    \pagestyle{plain}
    \def\@zoznamUv{\'{U}vod}
    \cleardoublepage
    \phantomsection
    \addcontentsline{toc}{section}{\@zoznamUv}
    \section*{\@zoznamUv}
    \noindent \input{#1}
}

\newcommand\FEIcore[1]{
    \let\stdsection\section
    \renewcommand\section{\newpage\stdsection}  
    \newpage
    \input{#1}
}

\newcommand\FEIconclusion[1]{   
    \newpage
        \def\@zoznamZa{Z\'{a}ver}
    \cleardoublepage
    \phantomsection 
    \addcontentsline{toc}{section}{\@zoznamZa}
    \section*{\@zoznamZa}
    \noindent \noindent \input{#1}
}



\newcommand\FEIbibliography{    
    \newpage
    \cleardoublepage
    \phantomsection 
	\def\@zoznamBib{Literatúra}
    \addcontentsline{toc}{section}{\@zoznamBib}
    \renewcommand\refname{\@zoznamBib} 

    % below you can set padding in bibliography
    \setlength{\bibhang}{0pt}
    \printbibliography
}



\newcounter{listOfAppendix}

\def\@appendices{Dodatky}
\def\@appendix{Dodatok}


\newcommand\FEIlistOfAppendix{
    \newpage
    \renewcommand \thesection{\Alph{section}}
    \setcounter{section}{0}
    \setcounter{subsection}{0}
    \setcounter{subsubsection}{0}
    \renewcommand \thetable{\Alph{section}}
    \setcounter{table}{0}
    \counterwithin{table}{section}
    \setcounter{figure}{0}
    \counterwithin{figure}{section}
    \setcounter{algorithm}{0}
    \counterwithin{algorithm}{section}
    \setcounter{lstlisting}{0}
    \counterwithin{lstlisting}{section}
    \cleardoublepage
    \phantomsection 
    \addcontentsline{toc}{section}{\@appendices}
    \listofpriloh
}

\newcommand\FEIappendix[2]{ 
    \newpage
    \cleardoublepage    
    \phantomsection 
    \section{#1}    
    \noindent
    \pridajPrilohu{#1}  
    \noindent\input{#2}
}
        
\newlistof{priloh}{pri}{\@appendices}
\newcommand{\pridajPrilohu}[1]{
    \refstepcounter{priloh}
    \addcontentsline{pri}{priloh}{\protect\numberline{\Alph{section}} #1}\par
}
        

%PZ: hack - front/main/backmatter v article 
%http://tex.stackexchange.com/questions/154646/
%\makeatletter

\newcommand\frontmatter{%
    \clearpage
    %\@mainmatterfalse
}

\newcommand\mainmatter{%
    \cleardoublepage
    % \@mainmattertrue
}

\newcommand\backmatter{%
    \if@openright
    \cleardoublepage
    \else
    \clearpage
    \fi
    \renewcommand \thesection{\Alph{section}}
    \titleformat{\section}
    {\normalfont\Large\bfseries}{\@appendix{} \thesection:}{1ex}{}
    \setcounter{section}{0}
    \setcounter{subsection}{0}
    \setcounter{subsubsection}{0}
    % \@mainmatterfalse
}
%\makeatother
\endinput
