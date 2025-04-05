<template>
  <div class="animate-fadeIn">
    <div class="mb-12 text-center">
      <h1 class="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
        Career Roadmap Explorer
      </h1>
      <p class="text-lg max-w-3xl mx-auto text-neutral/80 mb-12 leading-relaxed">
        Explore interactive career roadmaps for various paths. These diagrams provide a visual guide to help you
        understand the journey from education to career success. Choose a career path below to get started!
      </p>
    </div>

    <!-- Loading indicator -->
    <div v-if="loading" class="flex justify-center items-center py-16">
      <div class="relative w-16 h-16">
        <div class="absolute top-0 left-0 w-full h-full border-4 border-primary/20 rounded-full"></div>
        <div class="absolute top-0 left-0 w-full h-full border-4 border-primary rounded-full animate-spin border-t-transparent"></div>
      </div>
    </div>

    <!-- Career path bubbles -->
    <div v-else-if="careerPaths.length > 0" class="bubble-container">
      <div
        v-for="(path, index) in careerPaths"
        :key="path.id"
        class="bubble-wrapper"
      >
        <BubbleButton
          :color="getBubbleColor(index)"
          size="large"
          @click="navigateToCareerPath(path.id)"
        >
          <div class="flex flex-col items-center justify-center h-full">
            <span class="font-semibold">{{ path.name }}</span>
            <span class="text-xs mt-1 opacity-80">Click to explore</span>
          </div>
        </BubbleButton>
      </div>
    </div>

    <!-- No career paths found message -->
    <div v-else class="py-16 text-center">
      <div class="max-w-md mx-auto p-8 bg-white rounded-lg shadow-md border border-gray-100">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-red-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p class="text-xl font-medium text-neutral">No career paths found. Please check the data directory.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 3rem;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.bubble-wrapper {
  animation: float 6s ease-in-out infinite;
  margin: 1rem;
  width: 240px; /* Fixed width to ensure equal sizing */
  height: 240px; /* Fixed height to ensure equal sizing */
}

.bubble-wrapper:nth-child(3n+1) {
  animation-delay: 0s;
}

.bubble-wrapper:nth-child(3n+2) {
  animation-delay: 2s;
}

.bubble-wrapper:nth-child(3n) {
  animation-delay: 4s;
}

@keyframes float {
  0% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-15px);
  }
  100% {
    transform: translateY(0px);
  }
}

@media (max-width: 768px) {
  .bubble-container {
    gap: 2rem;
  }

  .bubble-wrapper {
    margin: 0.5rem;
  }
}
</style>

<script setup>
import { ref, onMounted } from 'vue';

// Router
const router = useRouter();
const runtimeConfig = useRuntimeConfig();

// State
const careerPaths = ref([]);
const loading = ref(true);
const errorMessage = ref(null);

// Bubble colors and sizes
const bubbleColors = [
  '#3B82F6', // primary (blue)
  '#10B981', // secondary (emerald)
  '#8B5CF6', // accent (purple)
  '#F59E0B', // warning (amber)
  '#06B6D4' // info (cyan)
];

// Fetch career paths on component mount
onMounted(async () => {
  try {
    await fetchCareerPaths();
  } catch (error) {
    console.error('Error fetching career paths:', error);
    errorMessage.value = 'Failed to load career paths. Please try again later.';
  } finally {
    loading.value = false;
  }
});

// Fetch available career paths
async function fetchCareerPaths() {
  try {
    const response = await fetch('/api/career-paths');

    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    // Ensure we have an array
    if (data && !Array.isArray(data)) {
      careerPaths.value = [data];
    } else {
      careerPaths.value = data || [];
    }
  } catch (e) {
    console.error('Error fetching career paths:', e);
    careerPaths.value = [];
  }
}

// Get bubble color based on index
function getBubbleColor(index) {
  return bubbleColors[index % bubbleColors.length];
}

// Navigate to career path page
function navigateToCareerPath(pathId) {
  navigateTo(`/career-path/${pathId}`);
}
</script>
