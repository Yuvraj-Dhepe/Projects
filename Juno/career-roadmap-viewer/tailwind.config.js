/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './components/**/*.{js,vue,ts}',
    './layouts/**/*.vue',
    './pages/**/*.vue',
    './plugins/**/*.{js,ts}',
    './app.vue',
    './error.vue'
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',    // Modern blue
        secondary: '#10B981',  // Emerald green
        accent: '#8B5CF6',     // Purple
        neutral: '#1F2937',    // Dark gray
        info: '#06B6D4',       // Cyan
        success: '#22C55E',    // Green
        warning: '#F59E0B',    // Amber
        error: '#EF4444',      // Red
        'base-100': '#ffffff',
        'base-200': '#F9FAFB',
        'base-300': '#F3F4F6',
      },
      animation: {
        'fadeIn': 'fadeIn 0.5s ease-in-out',
        'bounce-slow': 'bounce 3s infinite',
        'pulse-slow': 'pulse 3s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
    },
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: [
      {
        light: {
          primary: '#3B82F6',    // Modern blue
          secondary: '#10B981',  // Emerald green
          accent: '#8B5CF6',     // Purple
          neutral: '#1F2937',    // Dark gray
          info: '#06B6D4',       // Cyan
          success: '#22C55E',    // Green
          warning: '#F59E0B',    // Amber
          error: '#EF4444',      // Red
          'base-100': '#ffffff',
          'base-200': '#F9FAFB',
          'base-300': '#F3F4F6',
        },
      },
    ],
  },
}
