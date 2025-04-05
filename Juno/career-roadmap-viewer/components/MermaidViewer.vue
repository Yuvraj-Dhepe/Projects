<template>
  <div class="mermaid-viewer">
    <!-- Loading state -->
    <div v-if="loading" class="flex justify-center items-center py-8">
      <div class="relative w-12 h-12">
        <div class="absolute top-0 left-0 w-full h-full border-4 border-secondary/30 rounded-full"></div>
        <div class="absolute top-0 left-0 w-full h-full border-4 border-secondary rounded-full animate-spin border-t-transparent"></div>
      </div>
    </div>

    <!-- Error state -->
    <div v-else-if="error" class="bg-red-50 border-l-4 border-red-500 p-4 mb-4 rounded-md">
      <div class="flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" class="text-red-500 flex-shrink-0 h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span class="text-red-700">{{ error }}</span>
      </div>
    </div>

    <!-- Mermaid diagram -->
    <div v-else>
      <!-- Image view -->
      <div class="flex flex-col items-center">
        <div class="max-w-full overflow-x-auto bg-white p-4 rounded-lg transition-all duration-300 hover:shadow-lg">
          <img
            :src="imageUrl"
            :alt="alt"
            class="max-w-full h-auto rounded-md"
            @error="handleImageError"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';

// Props
const props = defineProps({
  careerPath: {
    type: String,
    required: true
  },
  format: {
    type: String,
    default: 'png'
  },
  alt: {
    type: String,
    default: 'Career Roadmap Diagram'
  }
});

// State
const loading = ref(true);
const error = ref(null);

// Computed properties
const imageUrl = computed(() => {
  return `/api/diagrams/${props.careerPath}?format=${props.format}`;
});

// Methods
function handleImageError() {
  error.value = 'Failed to load diagram image. Please try again later.';
}

// Watch for changes in the career path
watch(() => props.careerPath, () => {
  loading.value = true;
  // Add a small delay to show loading state
  setTimeout(() => {
    loading.value = false;
  }, 500);
});

// Initialize on component mount
onMounted(() => {
  // Add a small delay to show loading state
  setTimeout(() => {
    loading.value = false;
  }, 500);
});
</script>

<style scoped>
.mermaid-viewer img {
  transition: transform 0.3s ease;
}

.mermaid-viewer img:hover {
  transform: scale(1.02);
}
</style>
