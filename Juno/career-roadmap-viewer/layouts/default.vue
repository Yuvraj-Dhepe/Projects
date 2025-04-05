<template>
  <div class="min-h-screen flex flex-col bg-white text-neutral">
    <!-- Header -->
    <header class="py-4 bg-primary text-white shadow-md">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between">
          <NuxtLink to="/" class="flex items-center space-x-2 text-xl font-bold">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span>Career Roadmap Explorer</span>
          </NuxtLink>
          <nav class="hidden md:flex space-x-6">
            <a href="#" class="hover:text-secondary transition-colors duration-200">Home</a>
            <a href="#" class="hover:text-secondary transition-colors duration-200">Explore</a>
            <a href="#" class="hover:text-secondary transition-colors duration-200">About</a>
          </nav>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main class="flex-grow container mx-auto px-4 py-8 animate-fadeIn">
      <slot />
    </main>

    <!-- Footer with animation -->
    <footer class="py-6 bg-primary text-white">
      <div class="container mx-auto px-4">
        <div class="flex justify-center">
          <!-- Animated wave -->
          <div class="w-full h-12 overflow-hidden">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" class="w-full animate-pulse-slow" preserveAspectRatio="none">
              <path fill="rgba(249, 200, 70, 0.3)" fill-opacity="1" d="M0,96L48,112C96,128,192,160,288,186.7C384,213,480,235,576,224C672,213,768,171,864,149.3C960,128,1056,128,1152,149.3C1248,171,1344,213,1392,234.7L1440,256L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
            </svg>
          </div>
        </div>
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
