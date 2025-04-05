<template>
  <div class="pagination-controls">
    <div class="flex justify-between items-center mt-8 mb-2">
      <button
        @click="prevPage"
        class="pagination-btn prev-btn"
        :disabled="currentPage === 1"
        :class="{ 'disabled': currentPage === 1 }"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span>{{ prevPageTitle || 'Previous' }}</span>
      </button>

      <div class="flex gap-3">
        <template v-for="page in totalPages" :key="page">
          <button
            @click="goToPage(page)"
            class="pagination-dot"
            :class="{ 'active': page === currentPage }"
            :aria-label="`Go to page ${page}`"
          >
            <span class="sr-only">{{ page }}</span>
          </button>
        </template>
      </div>

      <button
        @click="nextPage"
        class="pagination-btn next-btn"
        :disabled="currentPage === totalPages"
        :class="{ 'disabled': currentPage === totalPages }"
      >
        <span>{{ nextPageTitle || 'Next' }}</span>
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  currentPage: {
    type: Number,
    required: true
  },
  totalPages: {
    type: Number,
    required: true
  },
  pageTitles: {
    type: Array,
    default: () => []
  }
});

const emit = defineEmits(['update:currentPage']);

const prevPageTitle = computed(() => {
  if (props.currentPage > 1 && props.pageTitles[props.currentPage - 2]) {
    return props.pageTitles[props.currentPage - 2];
  }
  return null;
});

const nextPageTitle = computed(() => {
  if (props.currentPage < props.totalPages && props.pageTitles[props.currentPage]) {
    return props.pageTitles[props.currentPage];
  }
  return null;
});

function prevPage() {
  if (props.currentPage > 1) {
    emit('update:currentPage', props.currentPage - 1);
  }
}

function nextPage() {
  if (props.currentPage < props.totalPages) {
    emit('update:currentPage', props.currentPage + 1);
  }
}

function goToPage(page) {
  emit('update:currentPage', page);
}
</script>

<style scoped>
.pagination-controls {
  margin-top: 2.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(0, 0, 0, 0.05);
}

.pagination-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 9999px;
  font-weight: 500;
  font-size: 0.875rem;
  color: #3B82F6;
  background-color: white;
  border: 1px solid rgba(59, 130, 246, 0.2);
  transition: all 0.2s ease;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.pagination-btn:hover {
  background-color: rgba(59, 130, 246, 0.05);
  border-color: rgba(59, 130, 246, 0.3);
  transform: translateY(-1px);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.pagination-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}

.pagination-dot {
  width: 0.75rem;
  height: 0.75rem;
  border-radius: 50%;
  background-color: rgba(59, 130, 246, 0.2);
  transition: all 0.2s ease;
}

.pagination-dot:hover {
  background-color: rgba(59, 130, 246, 0.4);
}

.pagination-dot.active {
  background-color: #3B82F6;
  transform: scale(1.2);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
</style>
