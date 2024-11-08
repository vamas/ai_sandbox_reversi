import pygame

from v1.agent import AgentType
from v1.gamestate import GameState
from v1.player import Player
from v1.position import Position

pygame.init()
pygame.font.init()

# Constants
# WINDOW_SIZE = 600  # Window dimensions (600x600 pixels)
WINDOW_SIZE = 600
FOOTER_HEIGHT = 100  # Footer height at the bottom of the screen

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + FOOTER_HEIGHT))
pygame.display.set_caption("Othello (Reversi)")
# Create a font object. You can specify the font type and size.
font = pygame.font.SysFont(None, 36)  # 'None' uses the default system font, and 36 is the size.


class GameManager:

    def __init__(self, agent_black, agent_white):
        self.game_state = GameState()
        self.agents = {Player.BLACK: agent_black, Player.WHITE: agent_white}

    @property
    def cell_size(self):
        """Getter method for color property."""
        return WINDOW_SIZE // self.game_state.grid_shape()

    def draw_board(self):
        screen.fill(GREEN)  # Fill the background with green
        # Draw grid lines
        for row in range(self.game_state.grid_shape()):
            for col in range(self.game_state.grid_shape()):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, GRAY, rect, 1)  # Draw the cell borders
                # Highlight legal moves
                if Position(row, col) in self.game_state.legal_moves:
                    pygame.draw.rect(screen, (34, 166, 34), rect)  # Highlight with a different color (e.g., bright green)

    def draw_pieces(self):
        board = self.game_state.board
        for row in range(self.game_state.grid_shape()):
            for col in range(self.game_state.grid_shape()):
                if self.game_state.board[row][col] == Player.BLACK:
                    pygame.draw.circle(screen, BLACK,
                                       (col * self.cell_size + self.cell_size // 2,
                                        row * self.cell_size + self.cell_size // 2),
                                       self.cell_size // 2 - 5)
                elif self.game_state.board[row][col] == Player.WHITE:
                    pygame.draw.circle(screen, WHITE,
                                       (col * self.cell_size + self.cell_size // 2,
                                        row * self.cell_size + self.cell_size // 2),
                                       self.cell_size // 2 - 5)

    def handle_click(self, pos):
        col = pos[0] // self.cell_size
        row = pos[1] // self.cell_size
        return self.game_state.make_move(Position(row, col))

    # Function to render text
    def render_text(self, text, position):
        # Render the text (True means antialiasing is on, and (255, 255, 255) is the text color - white)
        text_surface = font.render(text, True, (255, 255, 255))
        # Get the rectangle for positioning
        text_rect = text_surface.get_rect()
        # Set the position of the text
        text_rect.topleft = position
        # Blit the text surface onto the screen
        screen.blit(text_surface, text_rect)

    # Function to render text in the footer
    def render_footer(self, score=0):
        # Fill the footer area (clear it before rendering new text)
        pygame.draw.rect(screen, (0, 0, 0), (0, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE + FOOTER_HEIGHT))

        # Render variables (score and player turn in this case)
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        player_turn_text = font.render(f"Turn: {self.game_state.current_player}", True, (255, 255, 255))
        winner_text = font.render(f"Winner: {self.game_state.winner}", True, (255, 255, 255))

        # Blit the text onto the screen in the footer area
        screen.blit(score_text, (10, WINDOW_SIZE + 10))
        screen.blit(player_turn_text, (200, WINDOW_SIZE + 10))
        screen.blit(winner_text, (10, WINDOW_SIZE + 30))

    def run(self):
        """Main loop to run the game with a graphical interface."""
        running = True
        while not self.game_state.game_over and running:

            # Handle events, draw the board, display the current player, etc.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.agents[self.game_state.current_player].agent_type == AgentType.PLAYER:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Handle the click for the current player
                        self.handle_click(pygame.mouse.get_pos())
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.game_state.make_move(None)

            # Computer agent move
            if self.agents[self.game_state.current_player].agent_type == AgentType.COMPUTER:
                move = self.agents[self.game_state.current_player].get_best_move(self.game_state)
                move_info = self.game_state.make_move(move)

            # Draw the board and pieces
            self.draw_board()
            self.draw_pieces()
            self.render_footer()

            # Update the display
            pygame.display.flip()

        self.render_footer()
        pygame.display.flip()

        # Main game loop waiting for user input
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_input = False  # Exit loop if the user closes the window

                # Detect any key press or mouse button click
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting_for_input = False  # Exit loop if a key or mouse button is pressed

        pygame.quit()
        # Display the result when the game is over