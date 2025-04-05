<template>
  <div class="pagination-controls">
    <div class="flex justify-between items-center mt-6 mb-2">
      <button
        @click="prevPage"
        class="btn btn-outline btn-primary btn-sm gap-2"
        :disabled="currentPage === 1"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        {{ prevPageTitle || 'Previous' }}
      </button>

      <div class="flex gap-2">
        <template v-for="page in totalPages" :key="page">
          <button
            @click="goToPage(page)"
            class="btn btn-circle btn-sm"
            :class="page === currentPage ? 'btn-primary' : 'btn-outline btn-primary'"
          >
            {{ page }}
          </button>
        </template>
      </div>

      <button
        @click="nextPage"
        class="btn btn-outline btn-primary btn-sm gap-2"
        :disabled="currentPage === totalPages"
      >
        {{ nextPageTitle || 'Next' }}
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
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.btn-circle {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
}

.btn-primary {
  background-color: #4A6163;
  color: white;
}

.btn-outline.btn-primary {
  border-color: #4A6163;
  color: #4A6163;
}

.btn-outline.btn-primary:hover {
  background-color: #4A6163;
  color: white;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
