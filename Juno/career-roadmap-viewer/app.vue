<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, #__nuxt {
  font-family: 'Poppins', sans-serif;
  scroll-behavior: smooth;
  background-color: #F9FAFB;
  min-height: 100vh;
  margin: 0;
  padding: 0;
}

/* Ensure the app container takes full height */
#__nuxt {
  display: flex;
  flex-direction: column;
}

/* Smooth transitions for theme changes and animations */
* {
  transition: background-color 0.3s ease, color 0.3s ease, transform 0.3s ease, opacity 0.3s ease;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
