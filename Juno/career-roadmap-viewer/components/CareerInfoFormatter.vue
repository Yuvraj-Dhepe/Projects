<template>
  <div class="career-info-formatter">
    <div v-if="formattedContent" v-html="formattedContent"></div>
    <div v-else class="text-center py-8 text-neutral/70">
      No information available for this career path.
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue';
import DOMPurify from 'dompurify';

const props = defineProps({
  content: {
    type: String,
    default: ''
  }
});

const formattedContent = ref('');

// Process the content to add card styling
function processContent(content) {
  if (!content) return '';

  // Sanitize the HTML content
  let cleanContent = DOMPurify.sanitize(content);

  // Add special styling to "Why" sections
  cleanContent = cleanContent.replace(
    /<h2>(.*?why.*?matters|.*?benefits.*?|.*?important.*?)(.*?)<\/h2>/gi,
    '<h2 class="special-section">$1$2</h2>'
  );

  // Add check marks to list items in sections that follow "Why" headings
  cleanContent = cleanContent.replace(
    /(<h2 class="special-section">.*?<\/h2>\s*<ul>\s*<li)(>)(.*?)(<\/li>)/gi,
    '$1 class="check-item"$2<span class="check-mark">✓</span> $3$4'
  );

  // Add highlight boxes to certain paragraphs
  cleanContent = cleanContent.replace(
    /<p>(.*?provides.*?benefits|.*?valuable skills|.*?opportunities.*?advancement|.*?advantages.*?)<\/p>/gi,
    '<div class="highlight-box"><p>$1</p></div>'
  );

  return cleanContent;
}

// Watch for changes in the content prop
watch(() => props.content, (newContent) => {
  formattedContent.value = processContent(newContent);
}, { immediate: true });

// Process content on mount
onMounted(() => {
  formattedContent.value = processContent(props.content);
});
</script>

<style>
@import '~/assets/css/career-info-cards.css';

.career-info-formatter {
  width: 100%;
}

/* Check mark styling */
.check-item {
  padding-left: 1.75rem !important;
}

.check-mark {
  color: #5D8A66;
  font-weight: bold;
  position: absolute;
  left: 0;
  top: 0;
  font-size: 1.1rem;
}

/* Override the bullet point for check items */
.career-info-content .check-item::before {
  content: none !important;
}

/* Add some spacing between list items */
.career-info-content ul li {
  margin-bottom: 0.75rem;
}

/* Make the highlight boxes stand out more */
.highlight-box {
  background-color: #f0f9f0 !important;
  border-left: 4px solid #5D8A66 !important;
  padding: 1rem !important;
  margin: 1.5rem 0 !important;
  border-radius: 0 0.5rem 0.5rem 0 !important;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}
</style>
