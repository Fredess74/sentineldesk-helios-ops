/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        soc: {
          bg: '#060a13',
          panel: '#0d1424',
          card: '#111b30',
          border: '#1e2c48',
        },
      },
    },
  },
  plugins: [],
}
