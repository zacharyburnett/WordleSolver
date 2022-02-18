from functools import cache
from pathlib import Path
from typing import Collection, Dict, List

from pandas import Series
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

    letters = Series({letter: words.str.count(letter).sum() for letter in 'abcdefghijklmnopqurstuvwxyz'})

    return letters / letters.sum()


def word_letter_probabilities(words: Series) -> Series:
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

    words.reset_index(inplace=True, drop=True)
    return words


if __name__ == '__main__':
    in_place = {}
    out_of_place = {}
    not_in_word = []

    # get list of five-letter words sorted by their use in language
    words = five_letter_words_english()

    # sort word list by combined distinct probability score
    probabilities = word_letter_probabilities(words)
    words = Series(probabilities.sort_values(ascending=False).index)

    # main loop
    while True:
        words = word_choices(words, in_place, out_of_place, not_in_word)

        message = f'word list: {len(words)} words'
        if len(words) <= 50:
            print(f'{message} - {words.tolist()}')
        else:
            print(message)

        if len(words) <= 1:
            if len(words) == 1:
                print(f'the word should be "{words[0]}"')
            break

        # skip the first few most commonly-used words
        suggestion = words.iloc[0]
        print(f'suggestion: "{suggestion}"')

        green = input(f'capitalize green letters: ').strip()
        if len(green) == 0:
            green = suggestion

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

        yellow = input(f'capitalize yellow letters ("{word}"): ').strip()
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
