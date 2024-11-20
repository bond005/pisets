from dataclasses import dataclass
import difflib
import razdel
import numpy as np

def text_to_words(text: str) -> tuple[list[razdel.substring.Substring], list[bool]]:
    """
    Accepts a text, returns a list of tokens, where each token is either a word,
    or a punctuation mark. Additionally returns a boolean mask: is each token a
    punctuation mark? (True if does not contain alnum characters)
    """
    tokens = list(razdel.tokenize(text))
    for t in tokens:
        t.text = t.text.lower()
    is_a_punct = [all(not c.isalnum() for c in token.text) for token in tokens]
    return tokens, is_a_punct

@dataclass
class Match:
    """
    Represents a matching part between two lists:
    list1[start1:end1] matches list2[start2:end2]

    If self.len1 == self.len2, the fragments may be additionally be marked as
    equal or not equal (if not equal this is a replacement operation).
    """
    start1: int
    end1: int
    start2: int
    end2: int
    is_equal: bool

    def __post_init__(self):
        if self.is_equal:
            assert self.len1 == self.len2

    @property
    def len1(self) -> int:
        return self.end1 - self.start1

    @property
    def len2(self) -> int:
        return self.end2 - self.start2

    @property
    def is_replace(self) -> bool:
        return self.len1 > 0 and self.len2 > 0 and not self.is_equal

    @property
    def is_insert(self) -> bool:
        return self.len1 == 0

    @property
    def is_delete(self) -> bool:
        return self.len2 == 0

@dataclass
class CorrectionSuggestion:
    """
    A suggestion to correct some text in some place.
    """
    start_pos: int
    end_pos: int
    suggestion: str

def words_close_match(word1, word2) -> bool:
    return difflib.SequenceMatcher(None, word1, word2).ratio() >= 0.5

def compare(text1: str, text2: str) -> list[CorrectionSuggestion]:
    """
    Arguments:
    - text1: an ASR prediction
    - text2: another ASR prediction
    
    Returns a list of suggestions to replace, delete or insert something in the `text1`,
    based on the difference between both texts.

    Example:
    ```
    text1 = 'Раз, два, трии! Привет! Это "тестовый" текст. Корректор А. Кулакова.'
    text2 = 'ТРИ ПРИВЕТ ЭТО ЭЭ ТЕСТОВЫЙ ТЕКС'

    from asr.comparison import compare, visualize_correction_suggestions
    suggestions = compare(text1, text2)
    print(visualize_correction_suggestions(text1, suggestions))

    >>> {+Раз, два}, {трии|ТРИ}! Привет! Это {+ЭЭ} "тестовый" {текст|ТЕКС}. {+Корректор А. Кулакова}.
    ```
    """
    # parsing into words and punctuation marks
    tokens1, is_punct1 = text_to_words(text1)
    tokens2, is_punct2 = text_to_words(text2)

    # considering only words
    words1 = np.array(tokens1)[~np.array(is_punct1)].tolist()
    words2 = np.array(tokens2)[~np.array(is_punct2)].tolist()

    # get operations (delete, insert, replace, equal)
    matcher = difflib.SequenceMatcher(
        None,
        [t.text for t in words1],
        [t.text for t in words2],
        autojunk=False
    )
    orig_opcodes = matcher.get_opcodes()

    ops = [
        Match(start1, end1, start2, end2, is_equal=(op == 'equal'))
        for op, start1, end1, start2, end2 in orig_opcodes
    ]

    # now we have a list of Match-es between words1 and words2

    for _ in range(10):
        # we split some "replace" ops into two ops, such as 
        #    replace('aaaa bbb ccc', 'aaa') -> replace('aaaa', 'aaa') + delete('bbb ccc')
        new_ops: list[Match] = []
        for match in ops:
            start1, end1, start2, end2 = match.start1, match.end1, match.start2, match.end2
            if match.is_equal:
                new_ops.append(Match(start1, end1, start2, end2, is_equal=True))
            elif match.is_insert or match.is_delete:
                new_ops.append(Match(start1, end1, start2, end2, is_equal=False))
            elif match.is_replace:
                if words_close_match(words1[start1].text, words2[start2].text):
                    new_ops.append(Match(start1, start1 + 1, start2, start2 + 1, is_equal=False))
                    if end1 > start1 + 1 or end2 > start2 + 1:
                        new_ops.append(Match(start1 + 1, end1, start2 + 1, end2, is_equal=False))
                elif words_close_match(words1[end1 - 1].text, words2[end2 - 1].text):
                    if end1 - 1 > start1 or end2 - 1 >  start2:
                        new_ops.append(Match(start1, end1 - 1, start2, end2 - 1, is_equal=False))
                    new_ops.append(Match(end1 - 1, end1, end2 - 1, end2, is_equal=False))
                else:
                    new_ops.append(Match(start1, end1, start2, end2, is_equal=False))
        orig_ops = ops
        ops = new_ops
        if ops == orig_ops:
            break

    # consider only non-equal matches
    diffs = [op for op in ops if not op.is_equal]

    # get the positions in the original text, convert to correction suggestions
    suggestions: list[CorrectionSuggestion] = []

    for diff in diffs:
        # position
        if diff.start1 != diff.end1:
            text1_start_pos = words1[diff.start1].start
            text1_end_pos = words1[diff.end1 - 1].stop
        else:
            # suggestion to add
            if diff.end1 > 0:
                add_mode = 'append'
                pos = words1[diff.end1 - 1].stop
            else:
                add_mode = 'prepend'
                pos = words1[diff.end1].start
            text1_start_pos = pos
            text1_end_pos = pos
        
        # suggestion
        if diff.start2 == diff.end2:
            suggestion = ''
        else:
            text2_start_idx = words2[diff.start2].start
            text2_end_idx = words2[diff.end2 - 1].stop
            suggestion = text2[text2_start_idx:text2_end_idx]
            if diff.start1 == diff.end1:
                # suggestion to add
                if add_mode == 'append':
                    suggestion = ' ' + suggestion
                elif add_mode == 'prepend':
                    suggestion = suggestion + ' '
        
        suggestions.append(CorrectionSuggestion(text1_start_pos, text1_end_pos, suggestion))

    return suggestions

def visualize_correction_suggestions(text: str, suggestions: list[CorrectionSuggestion]) -> str:
    """
    Visualize suggestions in {brackets}.
    - {aaa|bbb} - suggest to replace aaa to bbb
    - {aaa} - suggest to remove aaa
    - {+aaa} - suggest to insert aaa (not present in `text`)
    
    Example:
    
    ```
    text1 = 'она советовала нам отнестись посему предмету к одному почтенному мужу'
    text2 = 'Она советовала нам отнести и спасену предмету к одному почтиному мужу.'
    suggestions = compare(text1, text2)
    print(visualize_correction_suggestions(text1, suggestions))

    >>> 'она советовала нам {отнестись|отнести} {+и} {посему|спасену} предмету к одному {почтенному|почтиному} мужу'
    ```
    """
    result = ''
    for i, suggestion in enumerate(suggestions):
        start = suggestion.start_pos
        end = suggestion.end_pos
        prev_end = suggestions[i - 1].end_pos if i > 0 else None

        result += text[prev_end:start]

        hypothesis1 = text[start:end]
        hypothesis2 = suggestion.suggestion
        if len(hypothesis1) == 0:
            # suggestion to add
            visualized_suggestion = '{+' + hypothesis2.strip() + '}'
            if hypothesis2.startswith(' '):
                visualized_suggestion = ' ' + visualized_suggestion
            if hypothesis2.endswith(' '):
                visualized_suggestion = visualized_suggestion + ' '
        elif len(hypothesis2) == 0:
            # suggestion to remove
            visualized_suggestion = '{+' + hypothesis1 + '}'
        else:
            # suggestion to correct
            visualized_suggestion = '{' + hypothesis1 + '|' + hypothesis2 + '}'
        
        result += visualized_suggestion
    
    result += text[end:]

    return result