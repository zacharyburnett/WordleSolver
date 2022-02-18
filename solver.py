from functools import cache
from pathlib import Path
import random
from typing import Collection, Dict, List

from pandas import DataFrame, Series
import pooch


@cache
def five_letter_words_english() -> Series:
    """
    :return: a list of five-letter English words from Stanford
    """

    words_filename = pooch.retrieve(
        'https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt',
        known_hash='52a04f4fb860953c2a29c2769014bd8b12d090a19e7577a460a2a2586bd6d4ce',
    )

    return Series(Path(words_filename).read_text().splitlines(), dtype='string')


def letter_probabilities(words: Series) -> Series:
    """
    :param words: Series of strings
    :return: letters with their probability of use out of 1
    """

    if not isinstance(words, Series):
        words = Series(words, dtype='string')

    letters = Series(
        {letter: words.str.count(letter).sum() for letter in 'abcdefghijklmnopqurstuvwxyz'}
    )

    return letters / letters.sum()


def word_letter_scores(words: Series) -> Series:
    """
    :param words: Series of words
    :return: words with their probability of use of distinct letters
    """

    letters = letter_probabilities(words)

    weighted_indices = Series(0, index=words.values)
    for letter, probability in letters.items():
        weighted_indices += words.str.contains(letter).values * probability

    return weighted_indices / weighted_indices.sum()


def word_choices(
    words: Series,
    in_place: Dict[int, str] = None,
    out_of_place: Dict[int, List[str]] = None,
    not_in_word: Collection[str] = None,
) -> Series:
    """
    filter words based on parameters

    :param words: Series of strings
    :param in_place: mapping of indices within the word to a character that exists there within the word
    :param out_of_place: mapping of indices within the word to a list of characters that does NOT exist there, but still exists within the word
    :param not_in_word: list of characters that do NOT exist within the word
    :return: list of words fitting the given parameters
    """

    if not isinstance(words, Series):
        words = Series(words, dtype='string')
    if in_place is None:
        in_place = {}
    if out_of_place is None:
        out_of_place = {}
    if not_in_word is None:
        not_in_word = []

    in_word = list(in_place.values())
    for index, characters in out_of_place.items():
        in_word.extend(characters)

    for index, character in in_place.items():
        words = words[words.str.get(index) == character]
    for character in in_word:
        words = words[words.str.contains(character)]
    for character in not_in_word:
        threshold = in_word.count(character)
        words = words[words.str.count(character) <= threshold]
    for index, characters in out_of_place.items():
        for character in characters:
            words = words[words.str.get(index) != character]

    return words


if __name__ == '__main__':
    in_place = {}
    out_of_place = {}
    not_in_word = []

    # get list of five-letter words sorted by their use in language
    words = five_letter_words_english()

    # sort word list by combined distinct probability score
    words = DataFrame(
        {'word': words.values, 'letter_score': word_letter_scores(words).values},
        index=words.index,
    )

    # main loop
    while True:
        words = words.loc[
            word_choices(words['word'], in_place, out_of_place, not_in_word).index
        ]

        message = f'word list: {len(words)} words'
        if len(words) <= 200:
            print(f'{message} - {words["word"].tolist()}')
        else:
            print(message)

        if len(words) <= 1:
            if len(words) == 1:
                print(f'the word should be "{words.iloc[0]["word"]}"')
            break

        # skip the first few most commonly-used words
        highest_letter_score_word = words[
            words['letter_score'] == words['letter_score'].max()
        ].iloc[0]['word']
        most_used_word = words.iloc[0]['word']
        print(f'highest letter score word: "{highest_letter_score_word}"')
        print(f'random word: "{words.loc[random.choice(words.index), "word"]}"')
        print(f'most-used word: "{most_used_word}"')

        green = input(f'capitalize the green letters of your word choice: ').strip()
        if len(green) == 0:
            green = most_used_word

        word = green.lower()
        if len(green) > 0 and all(character.isupper() for character in green):
            print(f'the word is "{word}"')
            break

        in_word_indices = set((*in_place, *out_of_place))
        for index, letter in enumerate(green):
            if letter.isupper():
                letter = letter.lower()
                in_place[index] = letter
                in_word_indices.add(index)

        yellow = input(
            f'capitalize the yellow letters of your word choice ("{word}"): '
        ).strip()
        for index, letter in enumerate(yellow):
            if letter.isupper():
                letter = letter.lower()
                if index not in out_of_place:
                    out_of_place[index] = []
                out_of_place[index].append(letter)
                in_word_indices.add(index)

        for index, letter in enumerate(word):
            if index not in in_word_indices:
                not_in_word.append(letter.lower())
