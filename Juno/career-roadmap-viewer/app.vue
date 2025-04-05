<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html {
  font-family: 'Poppins', sans-serif;
  scroll-behavior: smooth;
}

/* Smooth transitions for theme changes and animations */
* {
  transition: background-color 0.3s ease, color 0.3s ease, transform 0.3s ease, opacity 0.3s ease;
}

.page-enter-active,
.page-leave-active {
  transition: all 0.3s;
}
.page-enter-from,
.page-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>
