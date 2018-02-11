TARGET = solutions

STYFILE = $(wildcard *.sty)


all: $(TARGET).pdf

$(TARGET).pdf: ${PDF_FIGURES} $(TARGET).bbl $(wildcard *.tex) $(STYFILES)
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex

$(TARGET).aux: $(TARGET).tex $(wildcard *.tex)
	pdflatex $(TARGET)


$(TARGET).bbl: $(TARGET).bib $(TARGET).aux
	pdflatex $(TARGET).tex
	bibtex $(TARGET)

tags: TAGS
.PHONY: tags TAGS

TAGS:
	etags `latex-process-inputs --list solutions.tex`



clean:
	rm -f $(TARGET).pdf
	rm -f $(TARGET).log
	rm -f $(TARGET).blg
	rm -f $(TARGET).aux
	rm -f $(TARGET).bbl
	rm -f $(TARGET).dvi
	rm -f $(TARGET).out
	rm -f $(TARGET).toc

cleaner:
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.blg
	rm -f *.bbl
	rm -f *.dvi
	rm -f *.out
	rm -f *.toc
