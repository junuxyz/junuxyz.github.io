/**
 * Unified Blog State Management
 * Simple, consistent state handling for all pages
 */

(function() {
    'use strict';

    // =====================================
    // UNIFIED STATE SYSTEM
    // =====================================
    
    const BlogState = {
        // Theme management - simplified
        theme: {
            current: localStorage.getItem('pref-theme') || 'auto',
            
            init() {
                this.apply();
                this.bindToggle();
                this.bindSystemChange();
            },
            
            apply() {
                const body = document.body;
                
                if (this.current === 'dark') {
                    body.classList.add('dark');
                } else if (this.current === 'light') {
                    body.classList.remove('dark');
                } else {
                    // Auto mode - follow system preference
                    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    body.classList.toggle('dark', prefersDark);
                }
                
                // Force style refresh
                this.forceRefresh();
            },
            
            toggle() {
                const body = document.body;
                if (body.classList.contains('dark')) {
                    this.current = 'light';
                    body.classList.remove('dark');
                } else {
                    this.current = 'dark';
                    body.classList.add('dark');
                }
                
                localStorage.setItem('pref-theme', this.current);
                this.forceRefresh();
            },
            
            bindToggle() {
                const themeToggle = document.getElementById('theme-toggle');
                if (themeToggle) {
                    themeToggle.addEventListener('click', () => this.toggle());
                }
            },
            
            bindSystemChange() {
                if (this.current === 'auto') {
                    window.matchMedia('(prefers-color-scheme: dark)')
                        .addEventListener('change', () => this.apply());
                }
            },
            
            forceRefresh() {
                // Force browser to recalculate styles
                document.body.style.display = 'none';
                document.body.offsetHeight; // Trigger reflow
                document.body.style.display = '';
            }
        },
        
        // Unified layout fixes
        layout: {
            init() {
                this.fixBackgrounds();
                this.ensureConsistency();
            },
            
            fixBackgrounds() {
                // Force correct backgrounds on all elements
                const elements = document.querySelectorAll([
                    'html', 'body', '.main', '.post-single', '.post-content',
                    '.list', '.archive-posts', '.terms', '.page', '.home',
                    '.header', '.nav', '.footer'
                ].join(','));
                
                elements.forEach(el => {
                    el.style.background = 'var(--bg)';
                    el.style.color = 'var(--text)';
                });
            },
            
            ensureConsistency() {
                // Ensure unified styling across all page types
                const isDark = document.body.classList.contains('dark');
                document.documentElement.style.setProperty('--bg', isDark ? '#1A1714' : '#FAF8F5');
                document.documentElement.style.setProperty('--text', isDark ? '#F5F0EA' : '#2A2622');
            }
        },
        
        // Simple scroll memory
        scroll: {
            save() {
                localStorage.setItem('scroll-' + window.location.pathname, window.pageYOffset);
            },
            
            restore() {
                const position = localStorage.getItem('scroll-' + window.location.pathname) || 0;
                setTimeout(() => window.scrollTo(0, position), 100);
            },
            
            init() {
                this.restore();
                window.addEventListener('beforeunload', () => this.save());
            }
        }
    };

    // =====================================
    // INITIALIZATION
    // =====================================
    
    function init() {
        BlogState.theme.init();
        BlogState.layout.init();
        BlogState.scroll.init();
        
        console.log('Unified blog state initialized');
    }

    // Initialize when ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})(); 