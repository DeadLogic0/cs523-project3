marios_nn.py
  modify the main for either making a new GA, printing stats of last GA, or replaying the last GA

  Default Setup: Program is currently setup to display the last GA run
  Warning: A black window will pop up for the GUI even if just collecting stats when nothing will
    be displayed. This can be turned off by commenting out lines(136-139,683).

  New GA:
    Function to call in main: genetic_algorithm()
    If you would like to save the generation change save_gens(line 96) to True
      the saved generations will be saved to directory>gen(i)>file_names(i)_1
      directory variable(line 98)
      file_names variable(line 97)
    to display the best of each gen on the GUI change displayBest(line 93) to True
  Printing stats of last GA:
    Last GA is specified by the directory(line 98) and file_names(line 97) variables
    Function to call in main:
      fitness_all_gen()
          prints out the fitness for every generation of the last GA
      gen_against_gens_stats(gen)
          prints out the fitness difference between the specified generation and gen(n) where
          n = 0-num_of_gen(line 72) and n != gen
      gen_against_gen0_stats()
          prints out the fitness difference between gen(n) and gen(0) where n = 1-num_of_gen(line 72)
  Replaying last GA:
    Last GA is specified by the directory(line 98) and file_names(line 97) variables
    Function to call in main:
      replay_last_GA()
          function will display the best of every generation for the last GA on the GUI
          To fight against the AI put the code:
                                                global player
                                                player = True
          The controls for controlling Mario will be the arrow keys(UP, DOWN, LEFT, RIGHT)
          To increase the duration of the matches change arena_max_duration(line 593) to a larger number
