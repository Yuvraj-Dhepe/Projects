<template>
  <div class="mermaid-viewer">
    <!-- Loading state -->
    <div v-if="loading" class="flex justify-center items-center py-8">
      <div class="loading loading-spinner loading-lg text-primary"></div>
    </div>
    
    <!-- Error state -->
    <div v-else-if="error" class="alert alert-error shadow-lg mb-4">
      <div>
        <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        <span>{{ error }}</span>
      </div>
    </div>
    
    <!-- Mermaid diagram -->
    <div v-else>
      <!-- Render modes -->
      <div class="tabs tabs-boxed mb-4 justify-center">
        <a 
          @click="activeTab = 'image'" 
          :class="['tab', activeTab === 'image' ? 'tab-active' : '']"
        >
          Image View
        </a>
        <a 
          @click="activeTab = 'interactive'" 
          :class="['tab', activeTab === 'interactive' ? 'tab-active' : '']"
        >
          Interactive View
        </a>
        <a 
          @click="activeTab = 'code'" 
          :class="['tab', activeTab === 'code' ? 'tab-active' : '']"
        >
          Mermaid Code
        </a>
      </div>
      
      <!-- Image view -->
      <div v-if="activeTab === 'image'" class="flex flex-col items-center">
        <div class="max-w-full overflow-x-auto bg-white p-4 rounded-lg shadow-card">
          <img 
            :src="imageUrl" 
            :alt="alt"
            class="max-w-full h-auto"
            @error="handleImageError"
          />
        </div>
      </div>
      
      <!-- Interactive view -->
      <div v-else-if="activeTab === 'interactive'" class="flex flex-col items-center">
        <div class="max-w-full overflow-x-auto bg-white p-4 rounded-lg shadow-card">
          <div ref="mermaidContainer" class="mermaid-container"></div>
        </div>
      </div>
      
      <!-- Code view -->
      <div v-else-if="activeTab === 'code'" class="flex flex-col items-center">
        <div class="mockup-code w-full max-w-4xl">
          <pre><code>{{ mermaidCode }}</code></pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';

// Props
const props = defineProps({
  careerPath: {
    type: String,
    required: true
  },
  format: {
    type: String,
    default: 'svg'
  },
  alt: {
    type: String,
    default: 'Career Roadmap Diagram'
  }
});

// State
const loading = ref(true);
const error = ref(null);
const mermaidCode = ref('');
const activeTab = ref('image');
const mermaidContainer = ref(null);
const imageUrl = ref('');

// Watch for changes in props
watch([() => props.careerPath, () => props.format], () => {
  if (props.careerPath) {
    loadDiagram();
  }
});

// Watch for changes in active tab
watch(activeTab, async (newTab) => {
  if (newTab === 'interactive' && mermaidCode.value) {
    await renderMermaid();
  }
});

onMounted(async () => {
  if (props.careerPath) {
    await loadDiagram();
  }
});

// Load the diagram
async function loadDiagram() {
  loading.value = true;
  error.value = null;
  
  try {
    // Set the image URL
    imageUrl.value = `/api/diagrams/${props.careerPath}?format=${props.format}`;
    
    // Fetch the Mermaid code
    const response = await fetch(`/api/mermaid-code/${props.careerPath}`);
    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.message || 'Failed to load Mermaid code');
    }
    
    mermaidCode.value = data.code || '';
    
    // If interactive tab is active, render the Mermaid diagram
    if (activeTab.value === 'interactive') {
      await renderMermaid();
    }
  } catch (err) {
    console.error('Error loading diagram:', err);
    error.value = err.message || 'Failed to load diagram';
  } finally {
    loading.value = false;
  }
}

// Render Mermaid diagram
async function renderMermaid() {
  if (!mermaidCode.value || !mermaidContainer.value) return;
  
  try {
    // Import mermaid dynamically
    const { default: mermaid } = await import('mermaid');
    
    // Initialize mermaid
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
    });
    
    // Clear the container
    mermaidContainer.value.innerHTML = '';
    
    // Render the diagram
    const { svg } = await mermaid.render('mermaid-diagram', mermaidCode.value);
    mermaidContainer.value.innerHTML = svg;
  } catch (err) {
    console.error('Error rendering Mermaid diagram:', err);
    error.value = 'Failed to render Mermaid diagram';
  }
}

// Handle image loading errors
function handleImageError() {
  error.value = 'Failed to load diagram image';
}
</script>

<style scoped>
.mermaid-container {
  width: 100%;
  overflow-x: auto;
}

.mermaid-container :deep(svg) {
  max-width: 100%;
  height: auto;
}
</style>
