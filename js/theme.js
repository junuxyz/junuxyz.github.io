// Theme toggle functionality
(function() {
    'use strict';
    
    const THEME_KEY = 'theme';
    const THEMES = {
        LIGHT: 'light',
        DARK: 'dark',
        AUTO: 'auto'
    };
    
    // Get current theme from localStorage or default to auto
    function getCurrentTheme() {
        return localStorage.getItem(THEME_KEY) || THEMES.AUTO;
    }
    
    // Apply theme to document
    function applyTheme(theme) {
        const html = document.documentElement;
        
        // Remove existing theme classes
        html.classList.remove('dark');
        
        if (theme === THEMES.DARK) {
            html.classList.add('dark');
        } else if (theme === THEMES.AUTO) {
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (prefersDark) {
                html.classList.add('dark');
            }
        }
        // THEMES.LIGHT doesn't need any additional classes
    }
    
    // Toggle to next theme
    function toggleTheme() {
        const currentTheme = getCurrentTheme();
        let nextTheme;
        
        switch (currentTheme) {
            case THEMES.LIGHT:
                nextTheme = THEMES.DARK;
                break;
            case THEMES.DARK:
                nextTheme = THEMES.AUTO;
                break;
            case THEMES.AUTO:
            default:
                nextTheme = THEMES.LIGHT;
                break;
        }
        
        localStorage.setItem(THEME_KEY, nextTheme);
        applyTheme(nextTheme);
    }
    
    // Initialize theme on page load
    function initTheme() {
        const currentTheme = getCurrentTheme();
        applyTheme(currentTheme);
        
        // Listen for system theme changes when in auto mode
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
            if (getCurrentTheme() === THEMES.AUTO) {
                applyTheme(THEMES.AUTO);
            }
        });
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initTheme();
            
            // Add click listener to theme toggle button
            const themeToggle = document.getElementById('theme-toggle');
            if (themeToggle) {
                themeToggle.addEventListener('click', toggleTheme);
            }
        });
    } else {
        initTheme();
        
        // Add click listener to theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
        }
    }
})(); 