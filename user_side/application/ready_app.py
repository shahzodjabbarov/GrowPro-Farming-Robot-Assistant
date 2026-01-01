import pygame
import sys
import os
import subprocess
import requests
import json
import threading
from datetime import datetime

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GrowPro - Robot Control Interface")

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
font_tiny = pygame.font.SysFont('Arial', 20)

# Weather API Configuration
WEATHER_API_KEY = "838165d8802f46d0bf4184429252305"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"

# Cache for pre-rendered surfaces
surface_cache = {}
text_cache = {}

def get_cached_surface(key, create_func, *args):
    """Get or create cached surface to avoid recreating expensive operations"""
    if key not in surface_cache:
        surface_cache[key] = create_func(*args)
    return surface_cache[key]

def get_cached_text(text, font, color):
    """Cache rendered text surfaces"""
    cache_key = (text, font, color)
    if cache_key not in text_cache:
        text_cache[cache_key] = font.render(text, True, color)
    return text_cache[cache_key]

# Load and cache background image
def load_app_background():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        background_path = os.path.join(script_dir, "app_back.jpg")
        background = pygame.image.load(background_path).convert()
        return pygame.transform.scale(background, (WIDTH, HEIGHT))
    except Exception as e:
        print(f"Could not load app_back.jpg: {e}")
        # Create fallback gradient background once
        background = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            color_value = int(50 + (y / HEIGHT) * 100)
            pygame.draw.line(background, (color_value, color_value + 20, color_value + 40), (0, y), (WIDTH, y))
        return background

# Load and cache crop images
def load_crop_images():
    crop_images = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    crops = {
        'strawberry': 'straw.jpg',
        'pumpkin': 'pump.jpg',
        'lettuce': 'lett.jpg'
    }
    
    for crop, filename in crops.items():
        try:
            image_path = os.path.join(script_dir, filename)
            image = pygame.image.load(image_path).convert()
            crop_images[crop] = image
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            fallback = pygame.Surface((150, 150))
            if crop == 'strawberry':
                fallback.fill((255, 100, 100))
            elif crop == 'pumpkin':
                fallback.fill((255, 165, 0))
            else:
                fallback.fill((100, 255, 100))
            crop_images[crop] = fallback
    
    return crop_images

# Pre-load all assets
APP_BACKGROUND = load_app_background()
CROP_IMAGES = load_crop_images()

# Weather functions (moved to background thread)
def get_current_weather():
    try:
        params = {"key": WEATHER_API_KEY, "q": "Incheon", "aqi": "no"}
        response = requests.get(f"{WEATHER_BASE_URL}/current.json", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching current weather: {e}")
        return None

def get_forecast_weather():
    try:
        params = {"key": WEATHER_API_KEY, "q": "Incheon", "days": 5, "aqi": "no", "alerts": "yes"}
        response = requests.get(f"{WEATHER_BASE_URL}/forecast.json", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching forecast weather: {e}")
        return None

def analyze_farming_conditions(forecast_data):
    warnings = []
    if not forecast_data:
        return ["Unable to fetch weather data"]
    
    try:
        for day in forecast_data['forecast']['forecastday']:
            date = day['date']
            day_data = day['day']
            
            if day_data['totalprecip_mm'] > 10:
                warnings.append(f"{date}: Heavy rain expected ({day_data['totalprecip_mm']:.1f}mm)")
            if day_data['maxwind_kph'] > 25:
                warnings.append(f"{date}: Strong winds expected ({day_data['maxwind_kph']:.1f} km/h)")
            if day_data['maxtemp_c'] > 30:
                warnings.append(f"{date}: High temperature ({day_data['maxtemp_c']:.1f}°C)")
            elif day_data['mintemp_c'] < 5:
                warnings.append(f"{date}: Low temperature ({day_data['mintemp_c']:.1f}°C)")
            if day_data['avghumidity'] > 80:
                warnings.append(f"{date}: High humidity ({day_data['avghumidity']:.0f}%)")
        
        if 'alerts' in forecast_data and forecast_data['alerts']['alert']:
            for alert in forecast_data['alerts']['alert']:
                warnings.append(f"ALERT: {alert['headline']}")
        
        if not warnings:
            warnings.append("Good weather conditions for strawberry farming!")
            
    except Exception as e:
        warnings.append(f"Error analyzing weather data: {e}")
    
    return warnings

# Optimized gradient creation
def create_gradient_surface(width, height, color1, color2):
    """Create gradient surface once and cache it"""
    surface = pygame.Surface((width, height))
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, i), (width, i))
    return surface

# Optimized button class
class Button:
    def __init__(self, x, y, width, height, color, hover_color, text, text_color=WHITE, font=font_medium, gradient_color2=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.font = font
        self.is_hovered = False
        self.gradient_color2 = gradient_color2
        
        # Pre-render surfaces
        self._cache_key = f"btn_{x}_{y}_{width}_{height}_{color}_{gradient_color2}"
        self._hover_cache_key = f"btn_h_{x}_{y}_{width}_{height}_{hover_color}_{gradient_color2}"
        
        # Pre-render text
        self.text_surface = get_cached_text(text, font, text_color)
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)
        
    def get_surface(self):
        """Get the appropriate button surface (normal or hovered)"""
        if self.gradient_color2:
            cache_key = self._hover_cache_key if self.is_hovered else self._cache_key
            if self.is_hovered:
                return get_cached_surface(cache_key, create_gradient_surface, 
                                        self.rect.width, self.rect.height, 
                                        self.hover_color, self.gradient_color2)
            else:
                return get_cached_surface(cache_key, create_gradient_surface, 
                                        self.rect.width, self.rect.height, 
                                        self.color, self.gradient_color2)
        else:
            # Create solid color surface
            cache_key = self._hover_cache_key if self.is_hovered else self._cache_key
            def create_solid_surface():
                surf = pygame.Surface((self.rect.width, self.rect.height))
                color = self.hover_color if self.is_hovered else self.color
                surf.fill(color)
                return surf
            return get_cached_surface(cache_key, create_solid_surface)
        
    def draw(self, screen):
        # Draw cached button surface
        button_surface = self.get_surface()
        screen.blit(button_surface, self.rect)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, self.rect, 4, border_radius=15)
        
        # Draw cached text
        screen.blit(self.text_surface, self.text_rect)
        
    def check_hover(self, pos):
        old_hover = self.is_hovered
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered != old_hover  # Return True if hover state changed
        
    def is_clicked(self, pos, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and 
                event.button == 1 and self.rect.collidepoint(pos))

# Optimized crop button class
class CropButton:
    def __init__(self, x, y, size, crop_name, image):
        self.rect = pygame.Rect(x, y, size, size)
        self.crop_name = crop_name
        self.image = pygame.transform.scale(image, (size, size))
        self.is_hovered = False
        
        # Pre-render text surfaces
        self.text_surface = get_cached_text(crop_name.capitalize(), font_medium, WHITE)
        self.text_rect = self.text_surface.get_rect(center=(self.rect.centerx, self.rect.bottom + 25))
        
        # Pre-render text background
        self.bg_surface = pygame.Surface((self.text_surface.get_width() + 20, 
                                        self.text_surface.get_height() + 10))
        self.bg_surface.fill(BLACK)
        self.bg_surface.set_alpha(180)
        
    def draw(self, screen):
        # Draw image
        screen.blit(self.image, self.rect)
        
        # Draw border
        border_width = 6 if self.is_hovered else 3
        pygame.draw.rect(screen, WHITE, self.rect, border_width, border_radius=10)
        
        # Draw text with background
        screen.blit(self.bg_surface, (self.text_rect.x - 10, self.text_rect.y - 5))
        screen.blit(self.text_surface, self.text_rect)
        
    def check_hover(self, pos):
        old_hover = self.is_hovered
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered != old_hover
        
    def is_clicked(self, pos, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and 
                event.button == 1 and self.rect.collidepoint(pos))

# Optimized text drawing
def draw_text_with_background(screen, text, font, text_color, bg_color, x, y, alpha=180):
    text_surface = get_cached_text(text, font, text_color)
    
    # Cache background surfaces too
    bg_key = f"bg_{text}_{bg_color}_{alpha}"
    if bg_key not in surface_cache:
        bg_surface = pygame.Surface((text_surface.get_width() + 20, text_surface.get_height() + 10))
        bg_surface.fill(bg_color)
        bg_surface.set_alpha(alpha)
        surface_cache[bg_key] = bg_surface
    
    bg_surface = surface_cache[bg_key]
    screen.blit(bg_surface, (x - 10, y - 5))
    screen.blit(text_surface, (x, y))
    return text_surface.get_width(), text_surface.get_height()

# Page class
class Page:
    def __init__(self):
        self.buttons = []
        self.crop_buttons = []
        self.needs_redraw = True
        
    def add_button(self, button):
        self.buttons.append(button)
        
    def add_crop_button(self, crop_button):
        self.crop_buttons.append(crop_button)
        
    def draw(self, screen):
        if self.needs_redraw:
            screen.blit(APP_BACKGROUND, (0, 0))
            self.needs_redraw = False
        
        for button in self.buttons:
            button.draw(screen)
        for crop_button in self.crop_buttons:
            crop_button.draw(screen)
            
    def handle_event(self, event, mouse_pos):
        hover_changed = False
        
        # Check for hover effects
        for button in self.buttons:
            if button.check_hover(mouse_pos):
                hover_changed = True
        for crop_button in self.crop_buttons:
            if crop_button.check_hover(mouse_pos):
                hover_changed = True
        
        if hover_changed:
            self.needs_redraw = True
            
        # Check for clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, button in enumerate(self.buttons):
                if button.is_clicked(mouse_pos, event):
                    return ('button', i)
            for i, crop_button in enumerate(self.crop_buttons):
                if crop_button.is_clicked(mouse_pos, event):
                    return ('crop', i)
        return None

# Weather page with background loading
class WeatherPage(Page):
    def __init__(self):
        super().__init__()
        self.current_weather = None
        self.forecast_weather = None
        self.farming_warnings = []
        self.last_update = None
        self.loading = False
        
        self.add_button(Button(50, 520, 150, 60, GREY, DARK_GREY, "Back", WHITE, font_small, (200, 200, 200)))
        
    def update_weather_data(self):
        if self.loading:
            return
            
        def fetch_weather():
            self.loading = True
            try:
                self.current_weather = get_current_weather()
                self.forecast_weather = get_forecast_weather()
                self.farming_warnings = analyze_farming_conditions(self.forecast_weather)
                self.last_update = datetime.now()
            except Exception as e:
                print(f"Error updating weather: {e}")
            finally:
                self.loading = False
                self.needs_redraw = True
        
        # Run weather fetch in background thread
        threading.Thread(target=fetch_weather, daemon=True).start()
    
    def draw(self, screen):
        super().draw(screen)
        
        draw_text_with_background(screen, "Weather Information - Incheon", 
                                font_large, WHITE, BLACK, WIDTH//2 - 200, 20)
        
        y_offset = 80
        
        if self.loading:
            draw_text_with_background(screen, "Loading weather data...", 
                                    font_medium, WHITE, BLACK, WIDTH//2 - 120, y_offset + 100)
            return
        
        if not self.current_weather:
            draw_text_with_background(screen, "Unable to fetch weather data", 
                                    font_medium, RED, BLACK, WIDTH//2 - 150, y_offset + 100)
            return
        
        try:
            current = self.current_weather['current']
            
            temp = f"{current['temp_c']:.1f}°C"
            rain_chance = f"{current.get('precip_mm', 0):.1f}mm"
            humidity = f"{current['humidity']}%"
            wind = f"{current['wind_kph']:.1f} km/h"
            
            today_text = f"Today: {temp}, Rain: {rain_chance}, Humidity: {humidity}, Wind: {wind}"
            draw_text_with_background(screen, today_text, font_medium, WHITE, (0, 100, 0), 50, y_offset)
            
            y_offset += 60
            
            draw_text_with_background(screen, "Farming Conditions Alert:", 
                                    font_medium, YELLOW, BLACK, 50, y_offset)
            y_offset += 40
            
            for i, warning in enumerate(self.farming_warnings[:8]):
                color = RED if "ALERT" in warning or "damage" in warning else WHITE
                if len(warning) > 70:
                    warning = warning[:67] + "..."
                
                draw_text_with_background(screen, f"• {warning}", font_small, color, BLACK, 50, y_offset)
                y_offset += 30
            
            if self.last_update:
                update_text = f"Last updated: {self.last_update.strftime('%H:%M:%S')}"
                draw_text_with_background(screen, update_text, font_tiny, GREY, BLACK, WIDTH - 200, HEIGHT - 100)
            
            # Contact info
            x_position, y_position = 400, 500
            draw_text_with_background(screen, "Contact US", font_tiny, WHITE, BLACK, x_position + 50, y_position)
            draw_text_with_background(screen, "Phone Number: 01076462869", font_tiny, WHITE, BLACK, x_position, y_position + 35)
            draw_text_with_background(screen, "Email: farmscouts2025@gmail.com", font_tiny, WHITE, BLACK, x_position, y_position + 67)
            
        except Exception as e:
            draw_text_with_background(screen, f"Error displaying weather: {str(e)}", 
                                    font_medium, RED, BLACK, 50, y_offset + 100)

# Create pages
def create_pages():
    # Page 1 - Main Menu
    page1 = Page()
    page1.add_button(Button(350, 200, 300, 80, DARK_GREEN, GREEN, "Field 1", WHITE, font_medium, (0, 255, 100)))
    page1.add_button(Button(350, 320, 300, 80, GREY, DARK_GREY, "Add Field", WHITE, font_medium, (200, 200, 200)))
    page1.add_button(Button(350, 440, 300, 80, GREY, DARK_GREY, "Add Field", WHITE, font_medium, (200, 200, 200)))
    
    # Page 2 - Mode Selection
    page2 = Page()
    page2.add_button(Button(350, 150, 300, 80, DARK_BLUE, BLUE, "Auto", WHITE, font_medium, (100, 150, 255)))
    page2.add_button(Button(350, 270, 300, 80, DARK_BLUE, BLUE, "Manual", WHITE, font_medium, (100, 150, 255)))
    page2.add_button(Button(350, 390, 300, 80, DARK_BLUE, BLUE, "Follow", WHITE, font_medium, (100, 150, 255)))
    page2.add_button(Button(50, 520, 150, 60, GREY, DARK_GREY, "Back", WHITE, font_small, (200, 200, 200)))
    
    # Page 3 - Crop Selection
    page3 = Page()
    button_size = 150
    start_x = (WIDTH - (3 * button_size + 2 * 100)) // 2
    
    page3.add_crop_button(CropButton(start_x, 200, button_size, "strawberry", CROP_IMAGES['strawberry']))
    page3.add_crop_button(CropButton(start_x + button_size + 100, 200, button_size, "pumpkin", CROP_IMAGES['pumpkin']))
    page3.add_crop_button(CropButton(start_x + 2 * (button_size + 100), 200, button_size, "lettuce", CROP_IMAGES['lettuce']))
    page3.add_button(Button(50, 520, 150, 60, GREY, DARK_GREY, "Back", WHITE, font_small, (200, 200, 200)))
    
    # Page 4 - Functions Menu
    page4 = Page()
    button_width, button_height = 240, 70
    spacing_x, spacing_y = 50, 80
    start_x = (WIDTH - (3 * button_width + 2 * spacing_x)) // 2
    start_y = (HEIGHT - (2 * button_height + spacing_y)) // 2 - 20
    
    page4.add_button(Button(start_x, start_y, button_width, button_height, 
                           (200, 0, 0), (255, 50, 50), "Ripeness Detection", WHITE, font_small, (255, 100, 100)))
    page4.add_button(Button(start_x + button_width + spacing_x, start_y, button_width, button_height, 
                           (255, 140, 0), (255, 180, 50), "Disease Detection", WHITE, font_small, (255, 200, 100)))
    page4.add_button(Button(start_x + 2 * (button_width + spacing_x), start_y, button_width, button_height, 
                           DARK_GREEN, GREEN, "Weed Detection", WHITE, font_small, (100, 255, 150)))
    
    second_row_start_x = (WIDTH - (2 * button_width + spacing_x)) // 2
    page4.add_button(Button(second_row_start_x, start_y + button_height + spacing_y, button_width, button_height, 
                           DARK_BLUE, BLUE, "Soil Moisture Check", WHITE, font_small, (100, 150, 255)))
    page4.add_button(Button(second_row_start_x + button_width + spacing_x, start_y + button_height + spacing_y, 
                           button_width, button_height, PURPLE, (200, 100, 200), "Info", WHITE, font_small, (255, 150, 255)))
    
    page4.add_button(Button(50, 520, 150, 60, GREY, DARK_GREY, "Back", WHITE, font_small, (200, 200, 200)))
    
    # Page 5 - Weather Info Page
    page5 = WeatherPage()
    
    return [page1, page2, page3, page4, page5]

def main():
    clock = pygame.time.Clock()
    pages = create_pages()
    current_page = 0
    selected_mode = ""
    selected_function = ""
    selected_crop = ""
    
    # Track if page content needs updating
    page_changed = True
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            button_clicked = pages[current_page].handle_event(event, mouse_pos)
            
            if button_clicked is not None:
                click_type, click_index = button_clicked
                
                if current_page == 0:
                    if click_type == 'button' and click_index == 0:
                        current_page = 1
                        page_changed = True
                
                elif current_page == 1:
                    if click_type == 'button':
                        if click_index == 0:
                            selected_mode = "Auto"
                            current_page = 2
                            page_changed = True
                        elif click_index == 1:
                            selected_mode = "Manual"
                            current_page = 2
                            page_changed = True
                        elif click_index == 2:
                            selected_mode = "Follow"
                            current_page = 2
                            page_changed = True
                        elif click_index == 3:
                            current_page = 0
                            page_changed = True
                
                elif current_page == 2:
                    if click_type == 'crop':
                        crop_names = ['strawberry', 'pumpkin', 'lettuce']
                        selected_crop = crop_names[click_index]
                        current_page = 3
                        page_changed = True
                    elif click_type == 'button' and click_index == 0:
                        current_page = 1
                        page_changed = True
                
                elif current_page == 3:
                    if click_type == 'button':
                        functions = ["Ripeness Detection", "Disease Detection", "Weed Detection", "Soil Moisture Check", "Info"]
                        if click_index < len(functions):
                            selected_function = functions[click_index]
                            
                            if selected_function == "Info":
                                pages[4].update_weather_data()
                                current_page = 4
                                page_changed = True
                            else:
                                # Launch appropriate script
                                script_map = {
                                    'Auto': {'strawberry': 's_screen.py', 'pumpkin': 'p_screen.py', 'lettuce': 'l_screen.py'},
                                    'Manual': {'strawberry': 's_man_screen.py', 'pumpkin': 'p_man_screen.py', 'lettuce': 'l_man_screen.py'},
                                    'Follow': {'strawberry': 's_screen.py', 'pumpkin': 'p_screen.py', 'lettuce': 'l_screen.py'}
                                }
                                
                                script_name = script_map.get(selected_mode, {}).get(selected_crop, 'screen_rep.py')
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                script_path = os.path.join(script_dir, script_name)
                                
                                try:
                                    subprocess.Popen([sys.executable, script_path])
                                    print(f"Launched {selected_function} for {selected_crop} in {selected_mode} mode using {script_name}")
                                except Exception as e:
                                    print(f"Error launching {script_name}: {e}")
                                running = False
                            
                        elif click_index == 5:
                            current_page = 2
                            page_changed = True
                
                elif current_page == 4:
                    if click_type == 'button' and click_index == 0:
                        current_page = 3
                        page_changed = True
        
        # Only redraw if necessary
        if page_changed or pages[current_page].needs_redraw:
            SCREEN.blit(APP_BACKGROUND, (0, 0))
            
            # Draw page-specific titles
            if current_page == 0:
                draw_text_with_background(SCREEN, "GrowPro Robot Control System", 
                                        font_large, WHITE, BLACK, WIDTH//2 - 220, 80)
                draw_text_with_background(SCREEN, "Select Your Driving Field", 
                                        font_medium, WHITE, BLACK, WIDTH//2 - 130, 130)
            
            elif current_page == 1:
                draw_text_with_background(SCREEN, "Field 1 - Select Operating Mode", 
                                        font_large, WHITE, BLACK, WIDTH//2 - 200, 50)
            
            elif current_page == 2:
                draw_text_with_background(SCREEN, f"Field 1 - {selected_mode} Mode", 
                                        font_large, WHITE, BLACK, WIDTH//2 - 150, 50)
                draw_text_with_background(SCREEN, "Select Your Crop Type", 
                                        font_medium, WHITE, BLACK, WIDTH//2 - 110, 100)
            
            elif current_page == 3:
                draw_text_with_background(SCREEN, f"{selected_crop.capitalize()} - {selected_mode} Mode", 
                                        font_large, WHITE, BLACK, WIDTH//2 - 150, 20)
                draw_text_with_background(SCREEN, "Select Function to Execute", 
                                        font_medium, WHITE, BLACK, WIDTH//2 - 130, 60)
            
            pages[current_page].draw(SCREEN)
            page_changed = False
            pages[current_page].needs_redraw = False
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()