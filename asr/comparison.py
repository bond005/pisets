from __future__ import annotations

from dataclasses import dataclass
import difflib
import razdel

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
    def from_text(cls, text: str) -> TokenizedText:
        tokens = [
            Substring(
                start=t.start,
                stop=t.stop,
                text=t.text.lower(),
                is_punct=all(not c.isalnum() for c in t.text)
            )
            for t in razdel.tokenize(text)
        ]
        return TokenizedText(text=text, tokens=tokens)

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
                tokenized_text_2.get_words()
            )
        )

    # def get_character_alignment(self) -> 
    
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

        return ops

    def get_correction_suggestions(self) -> list[CorrectionSuggestion]:
        """
        Returns a list of suggestions to replace, delete or insert something in the `text1`,
        based on the difference between both texts. Thus, this function is not symmetric,
        since output suggestions contain positions in the `text1`. Punctuation is not compared,
        which means that the punctiation from `text2` is never used.
        """
        words1 = self.text1.get_words()
        words2 = self.text2.get_words()
        diffs = [op for op in self.matches if not op.is_equal]

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
                suggestion = self.text2.text[text2_start_idx:text2_end_idx]
                if diff.start1 == diff.end1:
                    # suggestion to add
                    if add_mode == 'append':
                        suggestion = ' ' + suggestion
                    elif add_mode == 'prepend':
                        suggestion = suggestion + ' '
            
            suggestions.append(CorrectionSuggestion(text1_start_pos, text1_end_pos, suggestion))

        return suggestions
    
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

@dataclass
class CorrectionSuggestion:
    """
    A suggestion to correct some text in some place, by replacing `text[start_pos:end_pos]`
    with `suggestion`. If `start_pos == end_pos`, this is a suggestion to add a text in
    `start_pos` position.
    """
    start_pos: int
    end_pos: int
    suggestion: str

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
    if len(suggestions) == 0:
        return text
    
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
            visualized_suggestion = '{' + hypothesis1 + '}'
        else:
            # suggestion to correct
            visualized_suggestion = '{' + hypothesis1 + '|' + hypothesis2 + '}'
        
        result += visualized_suggestion
    
    result += text[end:]

    return result