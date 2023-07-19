# source with wildcard
SOURCES := $(wildcard markdown/*.md)
NBS := $(patsubst markdown/%.md, notebooks/%.ipynb, $(SOURCES))

# rule to run
notebooks/%.ipynb: markdown/%.md
	pandoc --resource-path=assets/ --embed-resources --standalone --wrap=none  $< -o $@

all: $(NBS)

clean: 
	rm -f $(NBS)

