// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: process.env.NODE_ENV === 'development' },

  // Enable Tailwind CSS
  modules: ['@nuxtjs/tailwindcss'],

  // Global CSS
  css: ['~/assets/css/career-info.css'],

  // Configure app
  app: {
    head: {
      title: 'Career Roadmap Explorer',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Interactive explorer for career roadmap diagrams to help students visualize their career paths' },
        { name: 'theme-color', content: '#4A6163' },
        { name: 'og:title', content: 'Career Roadmap Explorer' },
        { name: 'og:description', content: 'Interactive explorer for career roadmap diagrams to help students visualize their career paths' },
        { name: 'og:type', content: 'website' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
      ]
    },
    pageTransition: { name: 'page', mode: 'out-in' }
  },

  // Configure runtime config
  runtimeConfig: {
    public: {
      baseUrl: process.env.BASE_URL || 'http://localhost:3000'
    },
    // Server-side environment variables
    debugMode: process.env.DEBUG_MODE || 'false'
  },

  // Configure nitro server
  nitro: {
    preset: 'vercel',
    // Enable caching for better performance
    storage: {
      cache: {
        driver: 'memory',
        ttl: 60 * 60 * 24 // 24 hours
      }
    }
  },

  // Optimize build
  build: {
    transpile: ['mermaid']
  },

  // Optimize performance
  experimental: {
    payloadExtraction: true,
    renderJsonPayloads: true,
    treeshakeClientOnly: true
  },

  // Optimize image loading
  routeRules: {
    '/api/diagrams/**': { cache: { maxAge: 60 * 60 * 24 } } // Cache diagrams for 24 hours
  }
})
