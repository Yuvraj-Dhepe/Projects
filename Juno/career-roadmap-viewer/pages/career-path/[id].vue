<template>
  <div class="animate-fadeIn">
    <!-- Header with career path name -->
    <div class="mb-8 text-center">
      <h1 class="text-4xl font-bold mb-4 text-primary">
        {{ careerPathName }}
      </h1>
      <p class="text-lg max-w-3xl mx-auto text-neutral/80 mb-8">
        Explore the roadmap and detailed information for this career path.
      </p>
    </div>

    <!-- Loading indicator -->
    <div v-if="loading" class="flex justify-center items-center py-16">
      <div class="relative w-16 h-16">
        <div class="absolute top-0 left-0 w-full h-full border-4 border-secondary/30 rounded-full"></div>
        <div class="absolute top-0 left-0 w-full h-full border-4 border-secondary rounded-full animate-spin border-t-transparent"></div>
      </div>
    </div>

    <!-- Error message -->
    <div v-else-if="error" class="py-16 text-center">
      <div class="max-w-md mx-auto p-8 bg-white rounded-lg shadow-md border border-gray-100">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-red-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p class="text-xl font-medium text-neutral">{{ error }}</p>
        <button @click="goBack" class="mt-4 btn btn-primary">
          Go Back
        </button>
      </div>
    </div>

    <!-- Main content with floating bubbles -->
    <div v-if="!loading && !error" class="bubble-container" :class="{ 'expanded-view': expandedBubble }">
      <!-- Regular bubbles when none is expanded -->
      <div v-if="!expandedBubble" class="regular-bubbles">
        <!-- Information Bubble -->
        <div class="bubble-wrapper">
          <BubbleButton
            color="#4A6163"
            size="large"
            @click="toggleBubble('info')"
          >
            <div class="flex flex-col items-center justify-center h-full">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span class="font-semibold">Career Information</span>
              <span class="text-xs mt-1 opacity-80">Click to view</span>
            </div>
          </BubbleButton>
        </div>

        <!-- Diagram Bubble -->
        <div class="bubble-wrapper">
          <BubbleButton
            color="#F9C846"
            size="large"
            @click="toggleBubble('diagram')"
          >
            <div class="flex flex-col items-center justify-center h-full">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
              <span class="font-semibold">Career Roadmap</span>
              <span class="text-xs mt-1 opacity-80">Click to view</span>
            </div>
          </BubbleButton>
        </div>

        <!-- Home Bubble -->
        <div class="bubble-wrapper">
          <BubbleButton
            color="#2F4858"
            size="large"
            @click="goBack"
          >
            <div class="flex flex-col items-center justify-center h-full">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              <span class="font-semibold">All Career Paths</span>
              <span class="text-xs mt-1 opacity-80">Return to home</span>
            </div>
          </BubbleButton>
        </div>
      </div>

      <!-- Small bubbles row when one is expanded -->
      <div v-if="expandedBubble" class="small-bubbles-row">
        <!-- Information Bubble -->
        <div class="bubble-wrapper small">
          <BubbleButton
            color="#4A6163"
            size="small"
            @click="toggleBubble('info')"
            v-if="expandedBubble !== 'info'"
          >
          <div class="flex flex-col items-center justify-center h-full">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span class="font-semibold">Info</span>
          </div>
        </BubbleButton>
        </div>

        <!-- Diagram Bubble -->
        <div class="bubble-wrapper small">
          <BubbleButton
            color="#F9C846"
            size="small"
            @click="toggleBubble('diagram')"
            v-if="expandedBubble !== 'diagram'"
          >
          <div class="flex flex-col items-center justify-center h-full">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
            <span class="font-semibold">Roadmap</span>
          </div>
        </BubbleButton>
        </div>

        <!-- Home Bubble -->
        <div class="bubble-wrapper small">
          <BubbleButton
            color="#2F4858"
            size="small"
            @click="goBack"
          >
          <div class="flex flex-col items-center justify-center h-full">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            <span class="font-semibold">Home</span>
          </div>
        </BubbleButton>
        </div>
      </div>

      <!-- Expanded content area -->
      <div class="expanded-content-area mt-4">
        <!-- Information Content -->
        <div v-if="expandedBubble === 'info'" class="expanded-content bg-white rounded-lg shadow-md p-6 border border-gray-100">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-semibold text-primary flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              About {{ careerPathName }}
            </h2>
            <button @click="toggleBubble(null)" class="btn btn-circle btn-sm">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="career-info-content overflow-y-auto max-h-[70vh]">
            <PagedCareerInfo :content="careerPathInfo" />
          </div>
        </div>

        <!-- Diagram Content -->
        <div v-if="expandedBubble === 'diagram'" class="expanded-content bg-white rounded-lg shadow-md p-6 border border-gray-100">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-semibold text-primary flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
              {{ careerPathName }} Roadmap
            </h2>
            <button @click="toggleBubble(null)" class="btn btn-circle btn-sm">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="diagram-content relative">
            <MermaidViewer
              :career-path="careerPathId"
              format="png"
              :alt="`${careerPathName} Career Path`"
            />

            <!-- Download button -->
            <a
              :href="`/api/diagrams/${careerPathId}?format=png`"
              download
              class="absolute top-2 right-2 bg-secondary text-white p-2 rounded-full shadow-md hover:bg-secondary/80 transition-colors duration-200"
              title="Download Diagram"
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';

// Route and router
const route = useRoute();
const careerPathId = computed(() => route.params.id);

// State
const loading = ref(true);
const error = ref(null);
const careerPathInfo = ref(null);
const careerPathName = ref('');
const expandedBubble = ref(null);

// Fetch data on mount
onMounted(async () => {
  try {
    await Promise.all([
      fetchCareerPathInfo(),
      fetchCareerPathName()
    ]);
  } catch (err) {
    console.error('Error loading career path data:', err);
    error.value = 'Failed to load career path data. Please try again later.';
  } finally {
    loading.value = false;
  }
});

// Fetch career path information
async function fetchCareerPathInfo() {
  try {
    console.log('Fetching career path info for:', careerPathId.value);
    const response = await fetch(`/api/career-paths/${careerPathId.value}/info`);

    if (!response.ok) {
      throw new Error(`Failed to fetch career path info: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Career path info response:', data);

    if (data && data.html) {
      careerPathInfo.value = data.html;
    } else {
      console.warn('No HTML content found in career path info response');
      careerPathInfo.value = '<p>No information available for this career path.</p>';
    }
  } catch (err) {
    console.error('Error fetching career path info:', err);
    careerPathInfo.value = '<p>Failed to load career path information.</p>';
    throw err;
  }
}

// Fetch career path name
async function fetchCareerPathName() {
  try {
    const response = await fetch('/api/career-paths');

    if (!response.ok) {
      throw new Error(`Failed to fetch career paths: ${response.statusText}`);
    }

    const data = await response.json();
    const path = data.find(p => p.id === careerPathId.value);

    if (!path) {
      throw new Error('Career path not found');
    }

    careerPathName.value = path.name;
  } catch (err) {
    console.error('Error fetching career path name:', err);
    throw err;
  }
}

// Toggle expanded bubble
function toggleBubble(bubble) {
  expandedBubble.value = expandedBubble.value === bubble ? null : bubble;
}

// Go back to home page
function goBack() {
  navigateTo('/');
}
</script>

<style scoped>
.bubble-container {
  display: flex;
  justify-content: center;
  gap: 3rem;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  transition: all 0.5s ease;
}

.bubble-container.expanded-view {
  flex-direction: column;
  align-items: center;
}

.regular-bubbles {
  display: flex;
  justify-content: center;
  gap: 3rem;
  width: 100%;
}

.small-bubbles-row {
  display: flex;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 2rem;
  width: 100%;
  position: relative;
  z-index: 10;
}

.expanded-bubble-content {
  width: 90%;
  max-width: 900px;
  margin: 0 auto;
}

.bubble-wrapper {
  animation: float 6s ease-in-out infinite;
  transition: all 0.5s ease;
  z-index: 1;
  width: 240px;
  height: 240px;
}

.bubble-wrapper.small {
  width: 120px;
  height: 120px;
  margin-top: 0;
}

.bubble-wrapper:nth-child(1) {
  animation-delay: 0s;
}

.bubble-wrapper:nth-child(2) {
  animation-delay: 2s;
}

.bubble-wrapper:nth-child(3) {
  animation-delay: 1s;
}

.bubble-wrapper.expanded {
  animation: none;
  width: 90%;
  height: auto;
  max-width: 900px;
  z-index: 10;
}

.expanded-content {
  width: 100%;
  min-height: 400px;
  transition: all 0.3s ease;
}

.career-info-content {
  font-size: 1rem;
  line-height: 1.6;
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
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
  }

  .regular-bubbles {
    flex-direction: column;
    gap: 2rem;
  }

  .small-bubbles-row {
    flex-wrap: wrap;
    gap: 1rem;
  }

  .bubble-wrapper.small {
    width: 100px;
    height: 100px;
  }

  .expanded-bubble-content {
    width: 100%;
  }
}
</style>
