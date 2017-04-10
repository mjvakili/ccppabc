LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out toc lof lot
CHECK_RERUN =

all: main.pdf clean.pdf


clean.pdf: *.tex *.bib
	${LATEX} clean_manuscript
	bibtex clean_manuscript 
	${LATEX} clean_manuscript 
	( grep Rerun main.log && ${LATEX} clean_manuscript ) || echo "Done."
	( grep Rerun main.log && ${LATEX} clean_manuscript ) || echo "Done."



main.pdf: *.tex *.bib
	${LATEX} main 
	bibtex main 
	${LATEX} main 
	( grep Rerun main.log && ${LATEX} main ) || echo "Done."
	( grep Rerun main.log && ${LATEX} main ) || echo "Done."


clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, main.${suff})
