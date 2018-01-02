### NLP Experiments with Patent data from the USPTO ###

#### Getting the Data ####

The url [Reed Tech USPTO Data Sets](http://patents.reedtech.com/pgrbft.php)
contains zip files with patent data. Each zip file contains one 'xml' file with information from multiple patents. The files can be downloaded from the link by clicking on the file name:

![Downloading a zip File](https://github.com/philiplessner/Patent-NLP/blob/master/ipynb/patent-website.png)

These files have to be parsed into indivdual xml files for each patent. Instructions on how to do so are in

[01extract-xml.ipynb](https://github.com/philiplessner/Patent-NLP/blob/master/ipynb/01extract-xml.ipynb)

Here we extract only the Utility patent files and ignore Design and Plant patent files. Each file is named patent_number.xml. For storage efficiency, the files are compressed into a tar.gz archive.

#### Getting Text from the XML Files ####

In this notebook 

[02xml2text.ipynb](https://github.com/philiplessner/Patent-NLP/blob/master/ipynb/02xml2text.ipynb) 

plain text is extracted from the xml files. In this notebook, title and abstract textare extracted, but other fields could be chosen. Each title/abstract pair is written to a text file as a new line terminated string.

#### Getting Metadata  ####

The xml files contain a rich set of metadata. I've extracted

* Patent Number
* References Cited
* IPC Classification Codes

for each patent. These are written to plain text files in the same order as the title/abstracts for each patent. Code for extracting the metadata can be found at

[03metadata.ipynb](https://github.com/philiplessner/Patent-NLP/blob/master/ipynb/03metadata.ipynb)

#### Tokinizing the Text ####

Code for converting the text to tokens can be found at

[04docs2tokens.ipynb](https://github.com/philiplessner/Patent-NLP/blob/master/ipynb/04docs2tokens.ipynb)

Punctuation is removed and the text is lowercased and lemmatized.

The text is also converted to individual sentences so it can be used later to obtain word embeddings.
