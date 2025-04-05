<template>
  <div class="min-h-screen flex flex-col bg-gradient-to-br from-base-200 to-white text-neutral w-full">
    <!-- Modern Header with centered rounded bar -->
    <div class="w-full bg-gradient-to-r from-primary/90 via-primary to-primary/90 py-3 shadow-lg">
      <div class="container mx-auto px-4 flex justify-center">
        <div class="bg-white rounded-full shadow-md px-8 py-3 flex items-center justify-center">
          <NuxtLink to="/" class="flex items-center space-x-3 text-xl font-bold text-primary">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span>Career Roadmap Explorer</span>
          </NuxtLink>
        </div>
      </div>
    </div>

    <!-- Main content -->
    <main class="flex-grow container mx-auto px-4 py-8 animate-fadeIn">
      <slot />
    </main>
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
