import random


def select_random_word():
    """
    Selects a random word from a predefined list of words.

    Returns:
    str: A randomly selected word.
    """
    word_list = ["apple", "banana", "cherry", "dog", "elephant", "flower",
                 "giraffe", "hamburger", "icecream", "jacket"]
    result = random.choice(word_list)
    return result


def display_word(word, guessed_letters):
    """
    Displays the word with guessed letters filled in and unguessed letters as underscores.

    Args:
    word (str): The target word to guess.
    guessed_letters (list): List of letters guessed by the player.

    Returns:
    str: The word with guessed letters filled in.
    """
    result = ""
    for letter in word:
        if letter in guessed_letters:
            result += letter
        else:
            result += "_"
    return result


def hangman_game():
    """
    Executes the Hangman game.

    The player must guess letters in a randomly selected word until they either win or lose.
    """
    word_to_guess = select_random_word()
    guessed_letters = []
    max_attempts = len(word_to_guess)  # Maximum attempts before losing

    print("Welcome to Hangman!")
    print(
        f"Try to guess the word. You can make up to {max_attempts} wrong guesses.")

    display_word(word_to_guess, guessed_letters)
    print(f"Word: {display_word(word_to_guess, guessed_letters)}")

    for attempt in range(1, max_attempts + 1):
        guessed_letter = input(f"Attempt {attempt}: Guess a letter: ").lower()
        guessed_letters.append(guessed_letter)
        print(f"Word: {display_word(word_to_guess, guessed_letters)}")

        if display_word(word_to_guess, guessed_letters) == word_to_guess:
            print("You win!")
            break
        elif attempt == max_attempts:
            print("You lose!")

    return None


if __name__ == '__main__':
    hangman_game()
