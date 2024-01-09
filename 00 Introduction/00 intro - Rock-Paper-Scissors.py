def play_rock_paper_scissors(player_choice, comp_choice):
    """
    Play a round of the Rock-Paper-Scissors game.

    Parameters:
        player1_choice (str): The choice of Player 1. Should be one of 'rock', 'paper', or 'scissors'.
        player2_choice (str): The choice of Player 2. Should be one of 'rock', 'paper', or 'scissors'.

    Returns:
        str: The result of the round. It can be one of the following:
             - "Tie" if both players have the same choice.
             - "Player 1 wins" if Player 1's choice wins the round.
             - "Player 2 wins" if Player 2's choice wins the round.

    Example:
        >>> play_rock_paper_scissors('rock', 'paper')
        'Player 2 wins'
        >>> play_rock_paper_scissors('scissors', 'scissors')
        'Tie'

    """
    result = ""

    # Check if it's a tie
    if player_choice == comp_choice:
        result = "Tie"

    # Determine the winner based on the rules
    elif (
            (player_choice == 'rock' and comp_choice == 'scissors') or
            (player_choice == 'scissors' and comp_choice == 'paper') or
            (player_choice == 'paper' and comp_choice == 'rock')
    ):
        result = 'Player wins'
    else:
        result = 'Comp wins'  # Player 2 is the computer

    # Check if the player's choices are valid
    valid_choices = {'rock', 'paper', 'scissors'}
    if player_choice not in valid_choices or comp_choice not in valid_choices:
        result = "Invalid choice. Choose from 'rock', 'paper', or 'scissors'."

    return result


import random


def computer_makes_choice():
    """
    Generates the choise of the computer in the Rock-Paper-Scissors game.

    Parameters:
        none.

    Returns:
        str: The choice of the player. Should be one of 'rock', 'paper', or 'scissors'.

    Example:
        >>> computer_makes_choice()
        'rock'
        >>> computer_makes_choice()
        'scissors'
    """
    valid_choices = {'rock', 'paper', 'scissors'}

    # Generate computer's choice randomly
    result = random.choice(list(valid_choices))

    return result


def play_multiple_rounds(num_rounds):
    """
    Play multiple rounds of the Rock-Paper-Scissors game against the computer.

    Parameters:
        num_rounds (int): The number of rounds to play.

    Returns:
        None

    Example:
        >>> play_multiple_rounds(3)
        Enter your choice (rock/paper/scissors): rock
        Round 1: You chose rock. Computer chose paper. Computer wins

        Enter your choice (rock/paper/scissors): paper
        Round 2: You chose paper. Computer chose paper. Tie

        Enter your choice (rock/paper/scissors): scissors
        Round 3: You chose scissors. Computer chose rock. Computer wins
    """
    valid_choices = {'rock', 'paper', 'scissors'}

    # Implementation of the dialog with the player

    for round in range(1, num_rounds + 1):
        player_choice = input(
            f"Enter your choice (rock/paper/scissors): ").lower()
        computer_choice = computer_makes_choice()
        result = play_rock_paper_scissors(player_choice, computer_choice)
        print(
            f"Round {round}: You chose {player_choice}. Computer chose {computer_choice}. {result}")

    return None


if __name__ == '__main__':
    # Example usage:
    num_rounds = int(input("Enter the number of rounds to play: "))
    play_multiple_rounds(num_rounds)
