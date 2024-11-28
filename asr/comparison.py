from __future__ import annotations

import copy
from dataclasses import dataclass
import difflib
from typing import Iterable, Literal
import numpy as np
import razdel
from pymystem3 import Mystem
from tqdm.auto import tqdm

@dataclass
class Substring:
    """
    Intended to store information about where words or punctuation marks are located
    in a text.
    
    This class is an extension of razdel.substring.Substring to store additional flags.
    """
    start: int
    stop: int
    text: str
    is_punct: bool

@dataclass
class TokenizedText:
    """
    Stores text and positions of tokens (words and punctuation marks).

    Tokenization is performed using Razdel (tested for Ru and En). A token
    is considered a punctuation mark if it does not contain letters or digits.

    Example:
    ```
    tokenized = TokenizedText.from_text('Это "тестовый" текст. !!')
    tokenized.tokens

    >>> [Substring(start=0, stop=3, text='это', is_punct=False),
        Substring(start=4, stop=5, text='"', is_punct=True),
        Substring(start=5, stop=13, text='тестовый', is_punct=False),
        Substring(start=13, stop=14, text='"', is_punct=True),
        Substring(start=15, stop=20, text='текст', is_punct=False),
        Substring(start=20, stop=21, text='.', is_punct=True),
        Substring(start=22, stop=24, text='!!', is_punct=True)]

    tokenized.get_words()

    >>> [Substring(start=0, stop=3, text='это', is_punct=False),
        Substring(start=5, stop=13, text='тестовый', is_punct=False),
        Substring(start=15, stop=20, text='текст', is_punct=False)]
    ```
    """
    text: str
    tokens: list[Substring]

    def get_words(self) -> list[Substring]:
        """
        Returns a list of words (skips punctuation marks).
        """
        return [t for t in self.tokens if not t.is_punct]

    @classmethod
    def from_text(cls, text: str, dash_as_separator: bool = True) -> TokenizedText:
        orig_text = text
        if dash_as_separator:
            text = text.replace('-', ' ')
        tokens = [
            Substring(
                start=t.start,
                stop=t.stop,
                text=t.text.lower(),
                is_punct=all(not c.isalnum() for c in t.text)
            )
            for t in razdel.tokenize(text)
        ]
        return TokenizedText(text=orig_text, tokens=tokens)

@dataclass
class WordLevelMatch:
    """
    A dataclass variant of `difflib.SequenceMatcher` outputs. Represents a matching
    part between two lists: `list1[start1:end1]` matches `list2[start2:end2]`

    If self.len1 == self.len2, may be additionally be marked as equal or not equal
    match (if not equal this Match represents a replacement operation).

    Use case: usually indices in Match are word indices (not character indices).
    """
    start1: int
    end1: int
    start2: int
    end2: int
    is_equal: bool

    char_start1: int | None = None
    char_end1: int | None = None
    char_start2: int | None = None
    char_end2: int | None = None

    def __post_init__(self):
        assert self.len1 > 0 or self.len2 > 0
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
class MultipleTextsAlignment:
    """
    Stores text, divided into words, and a list of found matches between the words.

    In the following example, we have two texts:
    ```
    text_1 = 'Aaaa aa, bb-bb'
    text_2 = 'Aa bbbb cc cc!'
    ```

    We split them into words with `TokenizedText`, which uses Razdel library under the hood.
    `TokenizedText` keeps a list of tokens, each token is either a lower-case word, or a
    punctuation mark.
    ```
    tokenized_text_1 = TokenizedText.from_text(text_1)
    tokenized_text_2 = TokenizedText.from_text(text_2)
    print(tokenized_text_1.tokens, tokenized_text_2.tokens)

    >>> [
        Substring(start=0, stop=4, text='aaaa', is_punct=False),
        Substring(start=5, stop=7, text='aa', is_punct=False),
        Substring(start=7, stop=8, text=',', is_punct=True),
        Substring(start=9, stop=14, text='bb-bb', is_punct=False)
    ], [
        Substring(start=0, stop=2, text='aa', is_punct=False),
        Substring(start=3, stop=7, text='bbbb', is_punct=False),
        Substring(start=8, stop=10, text='cc', is_punct=False),
        Substring(start=11, stop=13, text='cc', is_punct=False),
        Substring(start=13, stop=14, text='!', is_punct=True)
    ]
    ```

    We then match only words (a method `TokenizedText.get_words()`) in both texts:
    ```
    word_matches=MultipleTextsAlighment.get_matches(
        tokenized_text_1.get_words(),
        tokenized_text_2.get_words()
    )
    print(word_matches)
    
    >>> [
        WordLevelMatch(start1=0, end1=1, start2=0, end2=0, is_equal=False),
        WordLevelMatch(start1=1, end1=2, start2=0, end2=1, is_equal=True),
        WordLevelMatch(start1=2, end1=3, start2=1, end2=2, is_equal=False),
        WordLevelMatch(start1=3, end1=3, start2=2, end2=4, is_equal=False)
    ]
    ```

    For example, consider the last `WordLevelMatch`. It means that words [3:3] in
    `tokenized_text_1.tokens` match words [2:4] in `tokenized_text_2.tokens`. Since
    the first span is empty, this means that the last two words "cc" and "cc" in the
    second text have no counterparts in the first text. As for the other matches:

    - The 1st match is a deletion (the word "aaaa" is present only in the first text)
    - The 2nd match is an equality (the word "aa" is present in both texts)
    - The 3rd match is a replacement (the word "bb-bb" is replaced by "bbbb")
    - The 4th match is an insertion (the words "cc cc" are present only in the second text)

    Now we can construct `MultipleTextsAlighment`:
    ```
    alignment = MultipleTextsAlignment(tokenized_text_1, tokenized_text_2, word_matches)
    ```

    Or we can get the same result from the original texts using `.from_strings()`:
    ```
    alignment = MultipleTextsAlignment.from_strings(text_1, text_2)
    ```

    Now we can obtain the corrections that the second text suggests when compared with the
    first text. Here the positions (`start_pos`, `end_pos`) are character positions in the
    original `text_1`.
    ```
    suggestions = alignment.get_correction_suggestions()
    print(suggestions)

    >>> [
        CorrectionSuggestion(start_pos=0, end_pos=4, suggestion=''),
        CorrectionSuggestion(start_pos=9, end_pos=14, suggestion='bbbb'),
        CorrectionSuggestion(start_pos=14, end_pos=14, suggestion=' cc cc')
    ]
    ```

    We can visualize them in brackets, so that we can see all the matches: the deletion,
    the equality, the replacement and the insertion:
    ```
    print(visualize_correction_suggestions(text_1, suggestions))

    >>> '{Aaaa} aa, {bb-bb|bbbb} {+cc cc}'
    ```

    NOTE: while this class keeps a list of `WordLevelMatch`, and each match `m` may be one of
    `m.is_equal`, `m.is_delete`, `m.is_insert` or `m.is_replace`, they do not directly correspond
    one-to-one to "delete", "insert" and "replace" operations in Word Error Rate (WER) metric.
    Example:

    ```
    print(MultipleTextsAlignment.from_strings('a b c', 'd e').matches)

    >>> [WordLevelMatch(start1=0, end1=3, start2=0, end2=2, is_equal=False)]
    ```

    We can see a single "replace" operation from 3 words to 2 words. However, in WER metric this
    will be considered as two "replace" and one "delete" operation. To calculate WER correctly,
    use `.wer` property.
    """
    text1: TokenizedText
    text2: TokenizedText
    matches: list[WordLevelMatch]

    @classmethod
    def from_strings(cls, text1: str, text2: str) -> MultipleTextsAlignment:
        return MultipleTextsAlignment(
            text1=(tokenized_text_1 := TokenizedText.from_text(text1)),
            text2=(tokenized_text_2 := TokenizedText.from_text(text2)),
            matches=MultipleTextsAlignment.get_matches(
                tokenized_text_1.get_words(),
                tokenized_text_2.get_words(),
            )
        )

    def get_uncertainty_mask(self) -> np.ndarray:
        is_certain = np.full(len(self.text1.get_words()), False)
        for match in self.matches:
            is_certain[match.start1:match.end1] = match.is_equal
        return ~is_certain

    def wer(
        self,
        max_insertions: int | None = 4,
        uncertainty_mask: np.ndarray = None,
    ) -> dict:
        """
        Calculates WER. `max_insertions` allows to make WER more robust by not penalizing
        too much insertions in a row (usually an oscillatory hallucinations of ASR model).

        TODO switch to n unique insertions
        """
        _max_insertions = float('inf') if max_insertions is None else max_insertions

        words1 = self.text1.get_words()
        words2 = self.text2.get_words()
        
        n_equal = sum([m.len1 for m in self.matches if m.is_equal])
        n_deletions = sum([m.len1 for m in self.matches if m.is_delete])
        n_insertions = sum([min(m.len2, _max_insertions) for m in self.matches if m.is_insert])
        n_replacements = 0

        # replace operations contrubute to n_deletions and n_insertions if len1 != len2
        for match in self.matches:
            if match.is_replace:
                if match.len1 > match.len2:
                    n_replacements += match.len2
                    n_deletions += match.len1 - match.len2
                elif match.len1 < match.len2:
                    n_replacements += match.len1
                    n_insertions += min(match.len2 - match.len1, _max_insertions)
                else:
                    n_replacements += match.len1
        
        assert n_equal + n_deletions + n_replacements == len(words1)
        if max_insertions is None:
            assert n_equal + n_insertions + n_replacements == len(words2)

        results = {'wer': (n_deletions + n_insertions + n_replacements) / len(words1)}

        if uncertainty_mask is not None:
            assert len(uncertainty_mask) == len(words2)
            uncertainty_mask = uncertainty_mask.astype(bool)

            certain_n_correct = 0
            certain_n_incorrect = 0
            uncertain_n_correct = 0
            uncertain_n_incorrect = 0

            for match in self.matches:
                mask = uncertainty_mask[match.start2:match.end2]
                if match.is_equal:
                    uncertain_n_correct += mask.sum()
                    certain_n_correct += (~mask).sum()
                elif (match.is_insert or match.is_replace):
                    uncertain_n_incorrect += mask.sum()
                    certain_n_incorrect += (~mask).sum()

        if uncertainty_mask is not None:
            results['certain_n_correct'] = certain_n_correct
            results['certain_n_incorrect'] = certain_n_incorrect
            results['uncertain_n_correct'] = uncertain_n_correct
            results['uncertain_n_incorrect'] = uncertain_n_incorrect
            results['certain_correctness_ratio'] = (
                certain_n_correct / (certain_n_correct + certain_n_incorrect)
            )
            results['uncertain_correctness_ratio'] = (
                uncertain_n_correct / (uncertain_n_correct + uncertain_n_incorrect)
            )

        return results
    
    @staticmethod
    def get_matches(
        words1: list[Substring],
        words2: list[Substring],
        diff_only: bool = False,
        improved_matching: bool = True,
    ) -> list[WordLevelMatch]:
        """
        Finds matching words (excluding punctuation) in two word lists. If `diff_only`,
        returns only non-equal matches: deletions, additions or changes.

        With `improved_matching=True`, performs postprocessing after `difflib.SequenceMatcher`
        to split of join some matches.
        """
        # get operations (delete, insert, replace, equal)
        difflib_opcodes: list[tuple[str, int, int, int, int]] = difflib.SequenceMatcher(
            None,
            [t.text for t in words1],
            [t.text for t in words2],
            autojunk=False
        ).get_opcodes()

        ops: list[WordLevelMatch] = [
            WordLevelMatch(start1, end1, start2, end2, is_equal=(op == 'equal'))
            for op, start1, end1, start2, end2 in difflib_opcodes
        ]

        # now we have a list of Match-es between words1 and words2

        if improved_matching:
            for _ in range(10):
                # improvements over plain SequenceMatcher
                ops, was_change1 = MultipleTextsAlignment._maybe_split_replace_ops(words1, words2, ops)
                ops, was_change2 = MultipleTextsAlignment._maybe_join_subsequent_ops(words1, words2, ops)

                if not was_change1 and not was_change2:
                    break
        
        if diff_only:
            # consider only non-equal matches
            ops = [op for op in ops if not op.is_equal]

        # set character positions for each WordLevelMatch
        for op in ops:
            if op.start1 != op.end1:
                op.char_start1 = words1[op.start1].start
                op.char_end1 = words1[op.end1 - 1].stop
            else:
                if op.end1 > 0:
                    op.char_start1 = op.char_end1 = words1[op.end1 - 1].stop
                else:
                    op.char_start1 = op.char_end1 = words1[op.end1].start

            if op.start2 != op.end2:
                op.char_start2 = words2[op.start2].start
                op.char_end2 = words2[op.end2 - 1].stop
            else:
                if op.end2 > 0:
                    op.char_start2 = op.char_end2 = words2[op.end2 - 1].stop
                else:
                    op.char_start2 = op.char_end2 = words2[op.end2].start

        return ops

    @staticmethod
    def _string_match_score(word1: str, word2: str) -> float:
        """
        How similar are two strings (character-wise)?
        """
        return difflib.SequenceMatcher(None, word1, word2).ratio()

    @staticmethod
    def _maybe_split_replace_ops(
        words1: list[Substring],
        words2: list[Substring],
        ops: list[WordLevelMatch],
    ) -> tuple[list[WordLevelMatch], bool]:
        """
        We try to split some "replace" ops into two ops, such as 
        replace('aaaa bbb ccc', 'aaa') -> replace('aaaa', 'aaa') + delete('bbb ccc')
        
        Returns
        - a new ops list
        - flag that is True if any changes were made
        """
        new_ops: list[WordLevelMatch] = []
        for match in ops:
            start1, end1, start2, end2 = match.start1, match.end1, match.start2, match.end2
            if not match.is_replace:
                new_ops.append(match)
            else:
                if MultipleTextsAlignment._string_match_score(words1[start1].text, words2[start2].text) > 0.5:
                    new_ops.append(WordLevelMatch(start1, start1 + 1, start2, start2 + 1, is_equal=False))
                    if end1 > start1 + 1 or end2 > start2 + 1:
                        new_ops.append(WordLevelMatch(start1 + 1, end1, start2 + 1, end2, is_equal=False))
                elif MultipleTextsAlignment._string_match_score(words1[end1 - 1].text, words2[end2 - 1].text) > 0.5:
                    if end1 - 1 > start1 or end2 - 1 >  start2:
                        new_ops.append(WordLevelMatch(start1, end1 - 1, start2, end2 - 1, is_equal=False))
                    new_ops.append(WordLevelMatch(end1 - 1, end1, end2 - 1, end2, is_equal=False))
                else:
                    new_ops.append(match)
        
        return new_ops, (ops != new_ops)

    @staticmethod
    def _maybe_join_subsequent_ops(
        words1: list[Substring],
        words2: list[Substring],
        ops: list[WordLevelMatch],
    ) -> tuple[list[WordLevelMatch], bool]:
        """
        We try to merge two subsequent ops, such as
        delete('no', '') + replace('thing', 'nothing') -> replace('no thing', 'nothing')
        
        Returns
        - a new ops list
        - flag that is True if any changes were made
        """
        new_ops: list[WordLevelMatch] = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if i == len(ops) - 1:
                # the last op, cannot merge with subsequent op
                new_ops.append(op)
                i += 1
                continue
            next_op = ops[i + 1]
            if op.end1 != next_op.start1 or op.end2 != next_op.start2:
                # ops are not close to each other
                new_ops.append(op)
                i += 1
                continue
            if op.is_equal and next_op.is_equal:
                # we usually shouldn't have two `.is_equal` ops in a row, but just in case
                new_ops.append(op)
                i += 1
                continue
            op_words1 = ' '.join(x.text for x in words1[op.start1:op.end1])
            op_words2 = ' '.join(x.text for x in words2[op.start2:op.end2])
            next_op_words1 = ' '.join(x.text for x in words1[next_op.start1:next_op.end1])
            next_op_words2 = ' '.join(x.text for x in words2[next_op.start2:next_op.end2])
            
            match_score = MultipleTextsAlignment._string_match_score(op_words1, op_words2)
            next_match_score = MultipleTextsAlignment._string_match_score(next_op_words1, next_op_words2)
            joint_match_score = MultipleTextsAlignment._string_match_score(
                (op_words1 + ' ' + next_op_words1).strip(),
                (op_words2 + ' ' + next_op_words2).strip()
            )

            if joint_match_score > max(match_score, next_match_score):
                # merging ops
                new_ops.append(WordLevelMatch(op.start1, next_op.end1, op.start2, next_op.end2, is_equal=False))
                i += 2  # skipping the next op, since we've already merged it
            else:
                new_ops.append(op)
                i += 1
            
        return new_ops, (ops != new_ops)
    
    def substitute(
        self,
        replace: Iterable[int] | None = None,
        show_in_braces: Iterable[int] | None = None,
        pref_first: Iterable[int] | None = None,
        pref_second: Iterable[int] | None = None,
    ) -> str:
        """
        This function can insert fragments from the second text to the first text,
        based on matches.

        Explanation. Let we have a `MultipleTextsAlignment` with a single non-equal match
        (difference):

        ```
        text1 = "aa bb! cc!"
        text2 = "aa bbb cc"
        al = MultipleTextsAlignment.from_strings(text1, text2)
        print([m for m in al.matches if not m.is_equal])
        >>> [WordLevelMatch(start1=1, end1=2, start2=1, end2=2, is_equal=False,
            char_start1=3, char_end1=5, char_start2=3, char_end2=6)]
        ```
        
        The difference `m = al.matches[1]` corresponds to a substring in both texts:
        1) A segment in the 1st test: `al.text1.text[m.char_start1:m.char_end1] == 'bb'`
        2) A segment in the 2nd text: `al.text2.text[m.char_start2:m.char_end2] == 'bbb'`.
        
        Based on this, we can cut out the segment from the 1st text, and replace it
        with the segment from the 2nd text. This is exactly what does the `substitute` method.
        The `replace` argument is a list of all differences to apply.

        ```
        print(al.substitute(replace=[1]))
        >>> "aa bbb! cc!"
        ```
        
        The `show_in_braces` is also a list of differences. It does not replace text parts, but
        visualize both variants in {braces}.
        - {aaa|bbb} - suggest to replace aaa to bbb
        - {aaa} - suggest to remove aaa
        - {+aaa} - suggest to insert aaa (not present in `text1`)

        ```
        text1 = 'она советовала нам отнестись посему предмету к одному почтенному мужу'
        text2 = 'Она советовала нам отнести и спасену предмету к одному почтиному мужу.'
        al = MultipleTextsAlignment.from_strings(text1, text2)
        al.substitute(show_in_braces=range(len(al.matches)))
        >>> 'она советовала нам {отнестись|отнести} {+и} {посему|спасену} предмету к одному {почтенному|почтиному} мужу'
        ```
        """
        text1 = self.text1.text
        text2 = self.text2.text

        replace = list(replace) if replace is not None else []
        show_in_braces = list(show_in_braces) if show_in_braces is not None else []

        pref_first = list(pref_first) if pref_first is not None else []
        pref_second = list(pref_second) if pref_second is not None else []
        # assert set(pref_first).intersection(set(pref_second)) == set()

        result = ''
        text1_idx = 0

        for op_idx, op in enumerate(self.matches):
            if op.is_equal:
                continue

            result += text1[text1_idx:op.char_start1]
            text1_idx = op.char_start1

            segment1 = text1[op.char_start1:op.char_end1]
            segment2 = text2[op.char_start2:op.char_end2]

            if op_idx in replace:
                fragment = segment2
                
            elif op_idx in show_in_braces:
                if len(segment1) == 0:
                    formatting = 'add'
                elif len(segment2) == 0:
                    formatting = 'remove'
                else:
                    formatting = 'correct'
                
                if op_idx in pref_first:
                    segment1 = '!' + segment1
                if op_idx in pref_second:
                    segment2 = '!' + segment2

                if formatting == 'add':
                    fragment = '{+' + segment2.strip() + '}'
                    if text1[op.char_start1] == ' ':
                        fragment = ' ' + fragment
                    else:
                        fragment = fragment + ' '
                elif formatting == 'remove':
                    fragment = '{' + segment1 + '}'
                else:
                    fragment = '{' + segment1 + '|' + segment2 + '}'
            
            else:
                fragment = segment1
                
            result += fragment
            text1_idx = op.char_end1
        
        result += text1[text1_idx:]

        return result


def _is_junk_word(word: str) -> bool:
    return word in ['вот', 'ага', 'и', 'а', 'ну', 'это']

def _is_junk_word_sequence(text: str) -> bool:
    return text in ['то есть', 'да то есть', 'это самое']

def _lemmatize(text: str) -> str:
    return ''.join(Mystem().lemmatize(text)).strip()  # here we need to join with '', not ' '

def _should_keep(
    alignment: MultipleTextsAlignment,
    diff: WordLevelMatch,
    skip_word_form_change: bool,
) -> bool:
    """
    A single diff variant of .filter_correction_suggestions().
    """
    words1: list[str] = [w.text for w in alignment.text1.get_words()[diff.start1:diff.end1]]
    words2: list[str] = [w.text for w in alignment.text2.get_words()[diff.start2:diff.end2]]

    joined1 = ' '.join(words1).lower().replace('ё', 'е')
    joined2 = ' '.join(words2).lower().replace('ё', 'е')
    
    if all([_is_junk_word(w) for w in words1]) and all([_is_junk_word(w) for w in words2]):
        # insertion, replacement or deletion of junk words
        return False

    if (
        (len(joined1) == 0 or _is_junk_word_sequence(joined1))
        and (len(joined2) == 0 or _is_junk_word_sequence(joined2))
    ):
        # insertion, replacement or deletion of junk words
        return False

    if diff.is_replace:
        if joined1 == joined2:
            # the same text
            return False
        if skip_word_form_change and _lemmatize(joined1) == _lemmatize(joined2):
            # different forms of the same words, skip according to `skip_word_form_change=True`
            return False

        ru_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        has_ru1 = ru_letters & set(joined1) != set()
        has_ru2 = ru_letters & set(joined2) != set()

        if has_ru1 and not has_ru2:
            # probably a transliteration or letters-to-digits conversion
            return False
        if has_ru2 and not has_ru1:
            # probably a transliteration or letters-to-digits conversion
            return False
    
    return True

def filter_correction_suggestions(
    alignment: MultipleTextsAlignment,
    skip_word_form_change: bool = False
) -> list[int]:
    """
    Arguments:
    - alignment: a `MultipleTextsAlignment` between base speech recognition predictions and
    additional predictions from another model.
    - skip_word_form_change: whether to skips word form changes

    Outputs:
    - Indices all non-equal matches, filtered by several heuristics. This is treated as
      suggestions to replace, delete or insert something in the `text1`, based on the
      difference between words in both texts. Punctuation is not compared, since
      `MultipleTextsAlignment` ignores punctuation.

    NOTE: currently is adapted for Ru language
    """
    return [
        i for i, op in enumerate(tqdm(alignment.matches, desc='Filtering suggestions'))
        if not op.is_equal and _should_keep(
            alignment=alignment,
            diff=op,
            skip_word_form_change=skip_word_form_change
        )
    ]