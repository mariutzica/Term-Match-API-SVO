"""
Author: Maria Stoica
Date Modified: 15 November 2018
Description: This module contains the classes and function definitions for
            creating an "Ontology Categorizer." This "categorizer" takes any
            term (word), determines its possible senses in WordNet,
            and then maps senses to the different SVO ontological categories.
            Terms may have multiple categories, in which case they may need to
            be broken down into component concepts, as necessary.
"""

import pandas as pd
import nltk
#nltk.download('wordnet')
#nltk.download('brown')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


# An ontology category is the building block of the ontology categorizer.
# It consists of
#       - a name: the label of the category
#       - synsets: a list of wordnet synsets that are root nodes (top level parents)
#                  of the subtrees that are approximately matched to the ontology category
#       - verb: a Boolean value, set to true for categorization by pos == Verb
#       - adj: a Boolean value, set to true for categorization by pos == Adjective
class OntologyCategory:
    """ A class that supports tree root synsets for ontological categories. """

    # Constructor
    def __init__(self, name = None, synset_list = None):

        # a simple flag to indicate an error was encountered
        error = 0

        # default name valur for the category is Anonymous
        if name is None:
            self.name = 'Anonymous'
        else:
            self.name = name

        # verbs fall in the category 'Process'
        self.verb = False
        if self.name == 'process':
            self.verb = True

        # adjectives fall in the category 'Attribute'
        self.adj = False
        if self.name == 'attribute':
            self.adj = True

        # initialize a set of synsets mapped to the category
        self.synsets = set()
        if not synset_list is None:
            # unpack dict input to list as necessary
            # if not list or dict, trigger an error
            if isinstance(synset_list, list):
                ssit = synset_list
            elif isinstance(synset_list, dict):
                ssit = synset_list.items()
            else:
                print('Invalid synset: must be list or dict.')
                error = 1

            # attempt to unpack the list of term--index pairs
            # trigger error if not valid input
            if error == 0:
                try:
                    for term, index in ssit:
                        for i in index:
                            self.add_synset(term, i)
                except:
                    print('Invalid synset contents.')

    # add a synset to the category by term name and WordNet index
    def add_synset(self, term, index):
        try:
            self.synsets.add((term, index, wordnet.synsets(term)[index]))
        except:
            print('Error: could not find syset {} for {}.'.format(index, term))

    # remove a synset category by term name and WordNet index
    def remove_synset(self, term, index):
        rem = [r for r in self.synsets if (r[0]==term) and (r[1]==index)]
        if len(rem) > 0:
            self.synsets.discard(rem[0])
        else:
            print('Error: could not remove synset {} for {} because entry not in OntologyCategory.'\
            .format(index, term))

    # print definitions of all synset nodes in the ontology category
    def print_defs(self):
        for name, index, ss in self.synsets:
            print(name, '\t' if len(name)>6 else '\t\t', \
                  ss.definition())

    # determine if a given term has hypernymy in this category (is subclass of)
    def is_hypernym_of(self,hypernym_tree):
        # helper function that returns true when term is a Verb
        def is_verb(term):
            return term.pos()=='v'
        # helper function that returns true when term is an Adjective
        def is_adj(term):
            return (term.pos()=='a') or (term.pos()=='s')

        # step through the synsets in the hypernym tree
        hyp = []
        for ss in hypernym_tree:
            # if the synset matches, add it to the list of subtrees
            for name, index, sss in self.synsets:
                if ss == sss:
                    hyp.append(name+'.'+str(index))
            # if synset is a verb, and verb identifies category, then add
            if self.verb:
                if is_verb(ss):
                    hyp.append('verb')
            # if synset is adjective and adj identifies category, then add
            if self.adj:
                if is_adj(ss):
                    hyp.append('adjective')
        # if at least one element found, add name of the category to result
        if hyp != []:
            hyp.append(self.name)
        return hyp

class OntologyCategorizer():
    """ A class that supports categorization of terms by OntologyCategory() """

    # Constructor
    def __init__(self, name = None, categories = None):

        # default ontology category name is Anonymous
        if name is None:
            self.name = 'Anonymous'
        else:
            self.name = name

        # add categories if any are passed to constructor
        self.categories = []
        if not categories is None and isinstance(categories,list):
            for cat in categories:
                self.add_category(cat)

    # add a category
    def add_category(self, cat):
        self.categories.append(OntologyCategory(cat[0],cat[1]))

    # return a category
    def get_category(self, cat):
        category = [c for c in self.categories if c.name==cat]
        if len(category)==0:
            print('Error: Category {} not found.'.format(cat))
            category = ''
        else:
            category = category[0]
        return category

    # return list of category names
    def get_categories(self):
        return [c.name for c in self.categories]

    # remove a scategory by name
    def remove_category(self, name):
        rem = [r for r in self.categories if r.name==name]
        if len(rem) > 0:
            self.categories.discard(rem[0])
        else:
            print('Error: could not remove {} because category not present.'.format(name))

    # categorize a term
    def categorize_term(self, term, cat = None):

        # helper function that unpacks a hypernym tree and extracts
        # all of the synsets along all hypernym paths
        def det_hypernym(tree):
            elements = []
            for h in tree:
                if isinstance(h, list):
                    elements.extend(det_hypernym(h))
                else:
                    elements.append(h)
            return elements

        # get hypernym tree for the desired term
        hyp = []
        hyp_tree = det_hypernym(term.tree(lambda s:s.hypernyms()))

        # if no category selected, look up all categories
        if cat is None:
            for cat in self.categories:
                hyp.extend(cat.is_hypernym_of(hyp_tree))
        else:
            hyp.extend(cat.is_hypernym_of(hyp_tree))
        return hyp

    # return true/false depending on whether a term belongs to a selected category
    def iscat_ss(self, term, category):
        cat = self.get_category(category)
        return self.categorize_term(term, cat)!=[]

    # Determine what categories a given terms' word senses belong to in the ontology
    #   oc. Returns a pandas DataFrame object consisting of word senses along with
    #   categories and source synset(s) for each category.
    def what_is(self, term):

        term_cat = pd.DataFrame()
        term_ss = wordnet.synsets(term)
        loc = 0
        # loop through all of the synsets representing the term
        for ss in term_ss:
            index = len(term_cat)
            term_cat.loc[index,'term']=term
            term_cat.loc[index,'wordnet_ss_index']=loc
            term_cat.loc[index,'definition']=ss.definition()
            term_cat.loc[index,'pos']=ss.pos()
            hyp = self.categorize_term(ss)
            for h in hyp:
                term_cat.loc[index,h]='yes'
            loc += 1
        return term_cat.fillna('no')

    # Determine whether the word senses of a term belong to a given category
    def is_cat(self, term, cat, out = 'long'):

        term_cat = pd.DataFrame()
        term_ss = wordnet.synsets(term)
        loc = 0
        for ss in term_ss:
            index = len(term_cat)
            term_cat.loc[index,'term']=term
            term_cat.loc[index,'wordnet_ss_index']=loc
            term_cat.loc[index,'definition']=ss.definition()
            term_cat.loc[index,'pos']=ss.pos()
            term_cat.loc[index,cat]= 'yes' if self.iscat_ss(ss,cat) else 'no'
            loc += 1
        if out == 'long':
            return term_cat
        elif term_cat.empty:
            return False
        else:
            return (term_cat[cat]=='yes').any()

# Initialize the Scientific Variabes Ontology categorizer
#       return: object of class OntologyCategorizer
def init_svo():
    process_synsets  = ['process', {'process':[1,5], 'act':[1,5,6], \
                                'action':[0,1,3,4], 'event':[0] } ]
    property_synsets = ['property', {'property':[1,3], 'attribute':[0,1] } ]
    quantity_synsets = ['quantity', {'quantity':[0,2], 'amount':[0,2], \
                                 'ratio':[0], 'quantitative_relation':[0], \
                                 'distance':[0] } ]
    object_synsets   = ['phenomenon', {'object':[0,2,3,4], 'system':[1,4,5], \
                               'phenomenon':[0], 'body':[0,3,8], 'matter':[2], \
                               'form':[2,3,5,6], 'biological_group':[0], \
                               'body_of_water':[0], 'part':[2] } ]
    state_synsets = [ 'state', {'condition':[0,1,2], 'state':[1,4]}  ]
    attr_synsets = [ 'attribute', {}  ]

    return OntologyCategorizer('svo',[process_synsets, property_synsets, quantity_synsets,\
                                object_synsets, state_synsets, attr_synsets ])

### Command line functionality ... NO ERROR CHECKING
if __name__ == "__main__":
    import sys
    svo = init_svo()
    if len(sys.argv) == 2:
        whatis = svo.what_is(sys.argv[1])
        print(sys.argv[1]+' has the following categories:')
        found = False
        cols = whatis.columns.values
        for _,row in whatis.iterrows():
            cats = []
            if 'phenomenon' in cols and (row['object']=='yes'):
                cats.append('object')
            if 'process' in cols and (row['process']=='yes'):
                cats.append('process')
            if 'property' in cols and (row['property']=='yes'):
                cats.append('property')
            if 'state' in cols and (row['state']=='yes'):
                cats.append('state')
            if cats != []:
                found = True
                print(row['pos']+'. '+row['definition'])
                print('\t'+', '.join(cats))
        if not found:
            print('\tnone')
    elif len(sys.argv) == 3:
        iscat = svo.is_cat(sys.argv[1],sys.argv[2])
        print('The following definitions of '+sys.argv[1]+' are '+sys.argv[2]+':')
        found = False
        for _, row in iscat.iterrows():
            if row[sys.argv[2]]=='yes':
                print(row['pos']+'. '+row['definition'])
                found = True
        if not found:
            print('\tnone')
