import inspect
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from spacy.lang.en import English
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def correlate_on_pandas(self, canonical: pa.Table, header: str, code_str: str, to_header: str=None, seed: int=None,
                            save_intent: bool=None, intent_order: int=None, intent_level: [int, str]=None,
                            replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Allows a Pandas Series method to be run against a Table column. Examples of code_str:
                "str.extract('([0-9]+)').astype('float')"
                "apply(lambda x: x[0] if isinstance(x, str) else None)"

        :param canonical: a pa.Table as the reference table
        :param header: the header for the target values to change
        :param code_str: a code string matching a Pandas Series method such as str or apply
        :param to_header: (optional) an optional name to call the column
        :param seed: (optional) the random seed. defaults to current datetime
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        to_header  = self._extract_value(to_header)
        if not isinstance(header, str) or header not in canonical.column_names:
            raise ValueError(f"The header '{header}' can't be found in the canonical headers")
        seed = seed if isinstance(seed, int) else self._seed()
        s_values = canonical.column(header).combine_chunks().to_pandas()
        s_values = eval(f's_values.{code_str}', globals(), locals())
        if isinstance(s_values, pd.DataFrame):
            s_values = s_values.iloc[:, 0]
        rtn_arr = pa.Array.from_pandas(s_values)
        to_header = to_header if isinstance(to_header, str) else header
        return Commons.table_append(canonical, pa.table([rtn_arr], names=[to_header]))

    def correlate_replace(self, canonical: pa.Table, header: str, pattern: str, replacement: str, is_regex: bool=None,
                          max_replacements: int=None, seed: int=None, to_header: str=None, save_intent: bool=None,
                          intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ For each string in target, replace non-overlapping substrings that match the given literal pattern
        with the given replacement. If max_replacements is given and not equal to -1, it limits the maximum
        amount replacements per input, counted from the left. Null values emit null.

        If is a regex then RE2 Regular Expression Syntax is used

        :param canonical:
        :param header: The name of the target string column
        :param pattern: Substring pattern to look for inside input values.
        :param replacement: What to replace the pattern with.
        :param is_regex: (optional) if the pattern is a regex. Default False
        :param max_replacements: (optional) The maximum number of strings to replace in each input value.
        :param to_header: (optional) an optional name to call the column
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        to_header  = self._extract_value(to_header)
        is_regex = is_regex if isinstance(is_regex, bool) else False
        _seed = seed if isinstance(seed, int) else self._seed()
        c = canonical.column(header).combine_chunks()
        is_dict = False
        if pa.types.is_dictionary(c.type):
            is_dict = True
            c = c.dictionary_decode()
        if is_regex:
            rtn_values = pc.replace_substring_regex(c, pattern, replacement, max_replacements=max_replacements)
        else:
            rtn_values = pc.replace_substring(c, pattern, replacement, max_replacements=max_replacements)
        if is_dict:
            rtn_values = rtn_values.dictionary_encode()
        to_header = to_header if isinstance(to_header, str) else header
        return Commons.table_append(canonical, pa.table([rtn_values], names=[to_header]))

    def text_to_chunks(self, canonical: pa.Table, header: str, chunk_size: int=None, seed: int=None, save_intent: bool=None,
                          intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """

        :param canonical:
        :param header: The name of the target string column
        :param chunk_size:
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        _seed = seed if isinstance(seed, int) else self._seed()
        pages_and_chunks = []
        for item in canonical.to_pylist():
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {"page_number": item["page_number"]}
                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk
                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters
                pages_and_chunks.append(chunk_dict)

        return pa.Table.from_pylist(pages_and_chunks)

    def page_to_text(self, canonical: pa.Table, header: str, text_size: int=None, seed: int=None, save_intent: bool=None,
                     intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None):
        """

        :param canonical:
        :param header: The name of the target string column
        :param text_size:
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        _seed = seed if isinstance(seed, int) else self._seed()
        nlp = English()
        nlp.add_pipe("sentencizer")
        pages_and_texts = []
        for item in canonical.to_pylist():
            item["sentences"] = list(nlp(item[header]).sents)
            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            # split sentences into chunks
            item["sentence_chunks"] = [item["sentences"][i:i + text_size] for i in range(0, len(item["sentences"]), text_size)]
            pages_and_texts.append(item)
        return pa.Table.from_pylist(pages_and_texts)
