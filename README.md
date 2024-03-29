# Term-Match-API-SVO-1.0.0
A simple term matching API for SVO version 1.0.0

This API is implemented using a very simple Django wrapper. This API has been tested on MacOS Mojave + Anaconda and CentOS 7 running Python 3.

Requires the following packages and their dependencies (versions tested are in parentheses although likely this API may work with older versions of these packages):

  - pandas (0.24.2)
  - SPARQLWrapper (1.8.2)
  - nltk (3.4)
  - django (2.2.1)
  
The following nltk resources must be downloaded once before first use (just type the commands as shown after importing nltk):
  - nltk.download('wordnet')
  - nltk.download('brown')
  - nltk.download('punkt')
  - nltk.download('averaged_perceptron_tagger')


To run locally, simply download all files in this repo, make sure dependencies are installed (creating a new environment is highly recommended), and run the following command:

```python manage.py runserver 0.0.0.0:8000```
