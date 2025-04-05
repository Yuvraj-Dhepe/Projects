<template>
  <div class="min-h-screen flex flex-col bg-base-100 text-base-content">
    <!-- Header -->
    <header class="navbar bg-gradient-to-r from-primary to-secondary text-primary-content shadow-lg">
      <div class="container mx-auto px-4">
        <div class="flex-1">
          <NuxtLink to="/" class="btn btn-ghost normal-case text-xl font-bold">
            <span class="text-2xl mr-2">🚀</span> Career Roadmap Explorer
          </NuxtLink>
        </div>
        <div class="flex-none">
          <!-- Theme toggle -->
          <button @click="toggleTheme" class="btn btn-ghost btn-circle">
            <svg v-if="isDarkTheme" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
            <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          </button>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main class="flex-grow container mx-auto px-4 py-8 animate-fadeIn">
      <slot />
    </main>

    <!-- Footer -->
    <footer class="footer footer-center p-6 bg-base-200 text-base-content">
      <div class="grid grid-flow-col gap-4">
        <a href="#" class="link link-hover">About</a>
        <a href="#" class="link link-hover">Contact</a>
        <a href="#" class="link link-hover">Help</a>
      </div>
      <div>
        <div class="grid grid-flow-col gap-4">
          <a><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="fill-current"><path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"></path></svg></a>
          <a><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="fill-current"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"></path></svg></a>
          <a><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="fill-current"><path d="M9 8h-3v4h3v12h5v-12h3.642l.358-4h-4v-1.667c0-.955.192-1.333 1.115-1.333h2.885v-5h-3.808c-3.596 0-5.192 1.583-5.192 4.615v3.385z"></path></svg></a>
        </div>
      </div>
      <div>
        <p>
          Career Roadmap Explorer - Created with
          <a href="https://nuxt.com" target="_blank" class="link link-primary">Nuxt.js</a> and
          <a href="https://mermaid.js.org/" target="_blank" class="link link-primary">Mermaid.js</a>
        </p>
      </div>
    </footer>
  </div>
</template>

<script setup>
const isDarkTheme = ref(false);

// Initialize theme from localStorage if available
onMounted(() => {
  if (process.client) {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      isDarkTheme.value = true;
      document.documentElement.setAttribute('data-theme', 'dark');
    }
  }
});

// Toggle between light and dark themes
function toggleTheme() {
  isDarkTheme.value = !isDarkTheme.value;
  const theme = isDarkTheme.value ? 'dark' : 'light';

  if (process.client) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }
}
</script>
