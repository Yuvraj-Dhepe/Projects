<template>
  <div class="animate-fadeIn">
    <div class="mb-8 text-center">
      <h1 class="text-4xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary text-transparent bg-clip-text inline-block">
        Career Roadmap Explorer
      </h1>
      <p class="text-lg max-w-3xl mx-auto">
        Explore interactive career roadmaps for various paths. These diagrams provide a visual guide to help you
        understand the journey from education to career success. Choose a career path below to get started!
      </p>
    </div>

    <!-- Career path selector -->
    <div class="card bg-base-200 shadow-card hover:shadow-card-hover transition-shadow mb-8 max-w-3xl mx-auto">
      <div class="card-body">
        <h2 class="card-title flex items-center mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          Choose a Career Path
        </h2>

        <div class="form-control">
          <div class="input-group w-full">
            <select v-model="selectedCareerPath" class="select select-bordered w-full focus:outline-primary">
              <option disabled value="">Select a career path</option>
              <option v-for="path in careerPaths" :key="path.id" :value="path.id">
                {{ path.name }}
              </option>
            </select>
            <button class="btn btn-primary" @click="scrollToDiagram" :disabled="!selectedCareerPath">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Diagram display -->
    <div v-if="selectedCareerPath" id="diagram-section" class="card bg-base-200 shadow-card mb-8 max-w-5xl mx-auto">
      <div class="card-body">
        <h2 class="card-title text-2xl font-bold mb-4 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
          </svg>
          {{ getCurrentCareerPathName() }}
        </h2>

        <!-- Format selector -->
        <div class="flex justify-end mb-4">
          <div class="btn-group">
            <button
              @click="viewMode = 'png'"
              :class="['btn btn-sm', viewMode === 'png' ? 'btn-primary' : 'btn-outline btn-primary']"
            >
              PNG
            </button>
            <button
              @click="viewMode = 'svg'"
              :class="['btn btn-sm', viewMode === 'svg' ? 'btn-primary' : 'btn-outline btn-primary']"
            >
              SVG
            </button>
          </div>
        </div>

        <!-- Mermaid Viewer Component -->
        <MermaidViewer
          :career-path="selectedCareerPath"
          :format="viewMode"
          :alt="`${getCurrentCareerPathName()} Career Path`"
        />
      </div>
    </div>

    <!-- Career path information -->
    <div v-if="selectedCareerPath && careerPathInfo" class="card bg-base-200 shadow-card max-w-4xl mx-auto mb-8">
      <div class="card-body">
        <h2 class="card-title flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          About this Career Path
        </h2>
        <div v-html="careerPathInfo" class="prose max-w-none"></div>
      </div>
    </div>

    <!-- No selection message -->
    <div v-if="!selectedCareerPath && careerPaths.length > 0" class="py-16 text-center">
      <div class="max-w-md mx-auto p-8 bg-base-200 rounded-lg shadow-card">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-primary mb-4 animate-bounce-slow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
        </svg>
        <p class="text-xl font-medium">Please select a career path from the dropdown above to view its roadmap.</p>
      </div>
    </div>

    <!-- No career paths found message -->
    <div v-if="!loading && careerPaths.length === 0" class="py-16 text-center">
      <div class="max-w-md mx-auto p-8 bg-base-200 rounded-lg shadow-card">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-error mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p class="text-xl font-medium">No career paths found. Please check the data directory.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';

const runtimeConfig = useRuntimeConfig();
const baseUrl = runtimeConfig.public.baseUrl;

// State
const careerPaths = ref([]);
const selectedCareerPath = ref('');
const viewMode = ref('svg'); // Default to SVG for better quality
const loading = ref(true);
const careerPathInfo = ref(null);
const errorMessage = ref(null);

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

// Watch for changes in selected career path
watch(selectedCareerPath, async () => {
  if (selectedCareerPath.value) {
    loading.value = true;
    try {
      await loadCareerPathInfo();
    } catch (error) {
      console.error('Error loading career path info:', error);
      errorMessage.value = 'Failed to load career path information. Please try again later.';
    } finally {
      loading.value = false;
    }
  }
});

// Fetch available career paths
async function fetchCareerPaths() {
  console.log('Fetching career paths...');
  try {
    // Use fetch directly instead of useFetch
    const response = await fetch('/api/career-paths');
    console.log('Response status:', response.status);

    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('API raw response:', data);
    console.log('API response type:', typeof data);
    console.log('API response is array:', Array.isArray(data));

    // Ensure we have an array
    if (data && !Array.isArray(data)) {
      console.warn('API response is not an array, converting to array');
      careerPaths.value = [data];
    } else {
      careerPaths.value = data || [];
    }

    console.log('Career paths set to:', careerPaths.value);

    // Auto-select first career path if available
    if (careerPaths.value.length > 0) {
      console.log('Auto-selecting first career path:', careerPaths.value[0].id);
      selectedCareerPath.value = careerPaths.value[0].id;
    } else {
      console.log('No career paths available to select');
    }
  } catch (e) {
    console.error('Error fetching career paths:', e);
    careerPaths.value = [];
  }
}

// Load career path information
async function loadCareerPathInfo() {
  if (!selectedCareerPath.value) return;

  try {
    const { data } = await useFetch(`/api/career-path-info/${selectedCareerPath.value}`);
    careerPathInfo.value = data.value?.content || null;
  } catch (error) {
    console.error('Error loading career path info:', error);
    careerPathInfo.value = null;
  }
}

// Get the formatted name of the current career path
function getCurrentCareerPathName() {
  const path = careerPaths.value.find(p => p.id === selectedCareerPath.value);
  return path ? path.name : '';
}

// Scroll to diagram section
function scrollToDiagram() {
  if (!selectedCareerPath.value) return;

  // Use setTimeout to ensure the DOM has updated
  setTimeout(() => {
    const diagramSection = document.getElementById('diagram-section');
    if (diagramSection) {
      diagramSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, 100);
}
</script>
