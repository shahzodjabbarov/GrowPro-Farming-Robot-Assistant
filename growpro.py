import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Control Interface")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
GREY = (150, 150, 150)
BLUE = (0, 0, 200)
DARK_GREEN = (0, 150, 0)
DARK_GREY = (100, 100, 100)
DARK_BLUE = (0, 0, 150)
RED = (200, 0, 0)
YELLOW = (200, 200, 0)
PURPLE = (150, 0, 150)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
CYAN = (0, 200, 200)

# Font
font_large = pygame.font.SysFont('Arial', 40)
font_medium = pygame.font.SysFont('Arial', 30)
font_small = pygame.font.SysFont('Arial', 24)

# Button class for easy creation and handling
class Button:
    def __init__(self, x, y, width, height, color, hover_color, text, text_color=BLACK, font=font_medium):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.font = font
        self.is_hovered = False
        
    def draw(self, screen):
        # Draw button with hover effect
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=10)  # Border
        
        # Draw text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                return True
        return False

# Page class to organize different screens
class Page:
    def __init__(self):
        self.buttons = []
        
    def add_button(self, button):
        self.buttons.append(button)
        
    def draw(self, screen):
        screen.fill(WHITE)
        for button in self.buttons:
            button.draw(screen)
            
    def handle_event(self, event, mouse_pos):
        # Check for hover effects
        for button in self.buttons:
            button.check_hover(mouse_pos)
            
        # Check for clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, button in enumerate(self.buttons):
                if button.is_clicked(mouse_pos, event):
                    return i
        return None

# Create pages
def create_pages():
    # Page 1 - Main Menu
    page1 = Page()
    page1.add_button(Button(300, 200, 300, 80, GREEN, DARK_GREEN, "Field 1"))
    page1.add_button(Button(300, 320, 300, 80, GREY, DARK_GREY, "Add Field"))
    page1.add_button(Button(300, 440, 300, 80, GREY, DARK_GREY, "Add Field"))
    
    # Page 2 - Mode Selection
    page2 = Page()
    page2.add_button(Button(300, 150, 300, 80, BLUE, DARK_BLUE, "Auto"))
    page2.add_button(Button(300, 270, 300, 80, BLUE, DARK_BLUE, "Manual"))
    page2.add_button(Button(300, 390, 300, 80, BLUE, DARK_BLUE, "Follow"))
    page2.add_button(Button(50, 600, 150, 60, GREY, DARK_GREY, "Back"))
    
    # Page 3 - Functions Menu
    page3 = Page()
    page3.add_button(Button(250, 100, 300, 80, RED, (150, 0, 0), "Ripeness Detection"))
    page3.add_button(Button(650, 100, 300, 80, ORANGE, (200, 100, 0), "Disease Detection"))
    page3.add_button(Button(250, 220, 300, 80, GREEN, DARK_GREEN, "Weed Detection"))
    page3.add_button(Button(650, 220, 300, 80, BLUE, DARK_BLUE, "Soil Moisture Check"))
    page3.add_button(Button(250, 340, 300, 80, YELLOW, (150, 150, 0), "Crop/Harvest Count"))
    page3.add_button(Button(650, 340, 300, 80, BROWN, (100, 50, 0), "New Seed"))
    page3.add_button(Button(50, 600, 150, 60, GREY, DARK_GREY, "Back"))
    
    # Page 4 - Function Execution Page
    page4 = Page()
    page4.add_button(Button(50, 600, 150, 60, GREY, DARK_GREY, "Back"))
    
    return [page1, page2, page3, page4]

def main():
    clock = pygame.time.Clock()
    pages = create_pages()
    current_page = 0
    selected_mode = ""
    selected_function = ""
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Handle button clicks
            button_clicked = pages[current_page].handle_event(event, mouse_pos)
            
            if button_clicked is not None:
                # Page 1 (Main Menu) navigation
                if current_page == 0:
                    if button_clicked == 0:  # Field 1 button
                        current_page = 1  # Go to mode selection
                
                # Page 2 (Mode Selection) navigation
                elif current_page == 1:
                    if button_clicked == 0:  # Auto button
                        selected_mode = "Auto"
                        current_page = 2  # Go to functions menu
                    elif button_clicked == 1:  # Manual button
                        selected_mode = "Manual"
                        current_page = 2  # Go to functions menu
                    elif button_clicked == 2:  # Follow button
                        selected_mode = "Follow"
                        current_page = 2  # Go to functions menu
                    elif button_clicked == 3:  # Back button
                        current_page = 0  # Return to main menu
                
                # Page 3 (Functions Menu) navigation
                elif current_page == 2:
                    if button_clicked == 0:  # Ripeness Detection
                        selected_function = "Ripeness Detection"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 1:  # Disease Detection
                        selected_function = "Disease Detection"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 2:  # Weed Detection
                        selected_function = "Weed Detection"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 3:  # Soil Moisture Check
                        selected_function = "Soil Moisture Check"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 4:  # Crop/Harvest Count
                        selected_function = "Crop/Harvest Count"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 5:  # New Seed
                        selected_function = "New Seed"
                        current_page = 3  # Go to execution page
                    elif button_clicked == 6:  # Back button
                        current_page = 1  # Return to mode selection
                
                # Page 4 (Function Execution) navigation
                elif current_page == 3:
                    if button_clicked == 0:  # Back button
                        current_page = 2  # Return to functions menu
        
        # Draw current page
        pages[current_page].draw(SCREEN)
        
        # Additional drawing for specific pages
        if current_page == 0:
            title = font_large.render("Robot Control Interface", True, BLACK)
            SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        elif current_page == 1:
            title = font_large.render("Field 1 - Select Mode", True, BLACK)
            SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 50))
        
        elif current_page == 2:
            title = font_large.render(f"Field 1 - {selected_mode} Mode", True, BLACK)
            subtitle = font_medium.render("Select Function", True, BLACK)
            SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 20))
            SCREEN.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 60))
        
        elif current_page == 3:
            title = font_large.render(f"Field 1 - {selected_mode} Mode", True, BLACK)
            subtitle = font_large.render(selected_function, True, BLACK)
            
            SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 50))
            SCREEN.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 120))
            
            # Display placeholder for function execution
            msg = font_medium.render(f"Running {selected_function} in {selected_mode} mode", True, BLACK)
            SCREEN.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT//2))
            
            # This is where you would call your robot's actual functions
            # Example: if selected_mode == "Auto" and selected_function == "Ripeness Detection":
            #              run_ripeness_detection_auto()
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()