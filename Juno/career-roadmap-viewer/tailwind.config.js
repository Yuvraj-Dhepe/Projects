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
        primary: '#4361ee',    // Bright blue
        secondary: '#7209b7',  // Purple
        accent: '#f72585',     // Pink
        neutral: '#2b2d42',    // Dark blue-gray
        'base-100': '#ffffff',
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
          primary: '#4361ee',
          secondary: '#7209b7',
          accent: '#f72585',
          neutral: '#2b2d42',
          'base-100': '#ffffff',
        },
        dark: {
          primary: '#4361ee',
          secondary: '#7209b7',
          accent: '#f72585',
          neutral: '#2b2d42',
          'base-100': '#1f2937',
        },
      },
      'light',
      'dark',
    ],
  },
}
