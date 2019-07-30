from django.shortcuts import render
from django.http import HttpResponse
import re
import json
from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper import JSON as sqjson
from nltk.corpus import wordnet
import nltk
from . import ontology_category as oc

import os
from django.conf import settings

svo = oc.init_svo()

# look up term in ontology, return its class(es) if exact match found
def search_ontology_for_class(term):
    sparql = SPARQLWrapper("http://sparql.geoscienceontology.org")
    sparql.setQuery("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                    SELECT ?entity ?class
                    WHERE {{ ?entity a ?class .
                           ?entity rdfs:label ?label .
                           FILTER regex(?label,"^{}$") .}}
                    """.format(term))
    sparql.setReturnFormat(sqjson)
    results = sparql.query().convert()

    data = []
    for result in results["results"]["bindings"]:
        c = result["class"]["value"].split('#')[1]
        if not c in data:
            data.append(c)

    return data

# look up peripheral term in ontology; at this point this is agnostic to how
# the term is connected to the variable, but in the future it will be expanded
# to weigh main components more heavily than context or reference components.
def search_ontology_vars_periph(term):
    sparql = SPARQLWrapper("http://sparql.geoscienceontology.org")
    sparql.setQuery("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX svu: <http://www.geoscienceontology.org/svo/svu#>

                    SELECT ?variable ?label ?varlabel
                    WHERE {{ ?variable a svu:Variable .
                           ?variable rdfs:label ?varlabel .
                           ?variable svu:subLabel ?label .
                           FILTER regex(?label,"^{}$") .}}
                    """.format(term))
    sparql.setReturnFormat(sqjson)
    results = sparql.query().convert()

    data = []
    varlabels = []
    for result in results["results"]["bindings"]:
        c = result["variable"]["value"].split('#')[1]
        l = result["varlabel"]["value"]
        if not c in varlabels:
            data.append([c,l])
            varlabels.append(c)

    return data

# search and return phrase concept classes and related variables
def search_phrase(phrase, depth=0):
    terms = []
    #Quick and dirty way to parse a phrase and extract nouns
    # should move this up as its in the recursion stack currently
    is_noun = lambda pos: pos[:2] == 'NN'
    # go through the search phrase term by term
    for term in phrase.split('_'):
        # get the classes of a term and the variables explicitly linked to that term
        term_classes = search_ontology_for_class(term)
        term_variables = search_ontology_vars_periph(term)
        terms.append({'term':term,'classes':term_classes,'variables':term_variables})

        #only go two levels deep
        if depth<2:
            # here only 'state' definitions are expanded
            # this will be applied to attribute and phenomenon definitions as well in the future
            is_state = svo.is_cat(term,'state',out='short')
            if is_state:
                # grab synsets found that pertain to 'state'
                cat = svo.is_cat(term,'state')
                cat = cat.loc[cat['state']=='yes']
                # grab all of the synsets for the term
                term_ss = wordnet.synsets(term)
                syn_phrase_results = []
                # loop through the matched state definitions
                for d in cat['wordnet_ss_index'].tolist():
                    # simple algorithm:
                    # 1. grab term definition from wordnet
                    # 2. tokenize and extract nouns from phrase
                    # 3. call this search function recursively on the nouns in the definition
                    # NOTE: need to rank & filter out nouns to speed up this process
                    phrase = term_ss[int(d)].definition()
                    tokenized = nltk.word_tokenize(phrase)
                    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
                    phrase = '_'.join(nouns)
                    syn_phrase_results.append(search_phrase(phrase,depth+1))
                terms.append({'term':term,'expansions':syn_phrase_results})
    return terms

# based on num_matches, determine ontological matches
def rank_matches(phrase_results, max_results=5):

    # helper recrusive function that assigns rank to matches
    # currently depth is over-penalized; this penalty should be relaxed
    def tokens(phrase_results, var = {}, num_expansions = 1, num_terms = 1):
        for p in phrase_results:
            if 'variables' in p.keys():
                for v in p['variables']:
                    if v[0] in var.keys():
                        var[v[0]][0] += 1/num_expansions/num_terms
                    else:
                        var[v[0]] = [ 1/num_expansions/num_terms, 0, v[1]]
                        if 'Phenomenon' in p['classes']:
                            var[v[0]][1] = 1/num_expansions/num_terms
            elif 'expansions' in p.keys():
                num_expansions += len(p['expansions'])
                for ex in p['expansions']:
                    num_terms += len(ex)
                    var = tokens(ex,var,num_expansions,num_terms)
        return var

    var = tokens(phrase_results[1])
    # calculate match rank - currently matched by fraction of terms matched
    # rationale: simpler veriables, which are parent variables to more granular/ specific variables
    # will have a better rank match
    # depth explored
    num_total = len(phrase_results[1])
    for key in var:
        # count number of key terms in variable match
        num_variable = len([a for a in key.split('_') if (a!='') and (a!='of')])
        # add attribute count
        num_variable += key.count('%7E')
        # number of matched terms
        num_matched = var[key][0]
        # fraction of terms found weighed at 3/4, fraction of terms in variable matched weighed at 1/4
        temp = 0.75 * (num_matched/num_total) + 0.25 * (num_matched/num_variable)
        # penalize by expansion/depth
        temp *= 0.4+0.6*var[key][1]
        # store rank and variable match
        var[key] = [round(temp,3),var[key][2]]
    # sort the results by match rank in descending order and return the top max_results results
    sorted_results = sorted(var.items(), key=lambda kv: kv[1])
    sorted_results = sorted_results[::-1]
    results = [{"IRI":x[0],"label":x[1][1],"matchrank":x[1][0]} for x in sorted_results]
    return results[0:max_results]

# Create your views here.
def index(request, foo):
    if re.match("^[A-Za-z_]*$", foo):
        # sometimes connection is reset by peer; two attempts
        tries = 0
        while tries < 2:
            try:
                var_match = search_phrase(foo)
                tries = 2
            except Exception as e:
                print(e)
                tries += 1
        output = rank_matches([foo,var_match])
        resp = HttpResponse(json.dumps({"results":output}))
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Methods"] = "GET"
    else:
        resp = HttpResponse(f'invalid input ... {foo}')
    return resp

# display instructions on API usage
def instructions(request):
    with open(os.path.join(settings.BASE_DIR, 'match_phrase/index.html'), 'r') as file:
        output = file.read()
    return HttpResponse(output)
