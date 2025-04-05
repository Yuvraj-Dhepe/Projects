<template>
  <div class="paged-career-info">
    <!-- Current page content -->
    <div v-if="processedPages.length > 0" class="page-content">
      <div v-html="processedPages[currentPage - 1]"></div>
    </div>
    <div v-else class="text-center py-8 text-neutral/70">
      No information available for this career path.
    </div>

    <!-- Pagination controls -->
    <PaginationControls
      v-if="processedPages.length > 1"
      v-model:currentPage="currentPage"
      :totalPages="processedPages.length"
      :pageTitles="pageTitles"
    />
  </div>
</template>

<script setup>
import { ref, watch, onMounted, computed } from 'vue';
import DOMPurify from 'dompurify';
import PaginationControls from '~/components/PaginationControls.vue';

const props = defineProps({
  content: {
    type: String,
    default: ''
  }
});

const currentPage = ref(1);
const processedPages = ref([]);
const pageTitles = ref([
  'Overview & Requirements',
  'Career Progression',
  'Organizations & Certifications',
  'Tips & Resources'
]);

// Process the content to organize into pages and sections
function processContent(content) {
  if (!content) return [];

  // Sanitize the HTML content
  let cleanContent = DOMPurify.sanitize(content);

  // Create a temporary element to parse the HTML
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = cleanContent;

  // Find all h2 sections (main sections in markdown)
  const sections = {};
  const h2Elements = tempDiv.querySelectorAll('h2');

  // Group content by section headings
  h2Elements.forEach(h2 => {
    const sectionName = h2.textContent.trim();
    sections[sectionName] = { heading: h2, content: extractSectionContent(h2) };
  });

  // Create pages based on our structure
  const pages = [
    createOverviewPage(sections),
    createProgressionPage(sections),
    createOrganizationsPage(sections),
    createTipsResourcesPage(sections)
  ];

  return pages;
}

// Extract all content for a section until the next h2
function extractSectionContent(h2Element) {
  let content = '';
  let currentElement = h2Element.nextElementSibling;

  while (currentElement && currentElement.tagName !== 'H2') {
    content += currentElement.outerHTML;
    currentElement = currentElement.nextElementSibling;
  }

  return content;
}

// Create the Overview & Requirements page
function createOverviewPage(sections) {
  let page = '<div class="career-page">';

  // Overview Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Overview</h2>';
  page += '<div class="card-container">';

  // Get overview content from sections
  const overviewContent = sections['Overview']?.content || '';

  page += `<div class="info-card">${overviewContent || '<p>No overview information available.</p>'}</div>`;
  page += '</div></div>'; // End of Overview Section

  // Education Requirements Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Education Requirements</h2>';
  page += '<div class="card-container">';

  // Get education requirements content from sections
  const educationContent = sections['Education Requirements']?.content || '';

  page += `<div class="info-card">${educationContent || '<p>No education requirements information available.</p>'}</div>`;
  page += '</div></div>'; // End of Education Requirements Section

  // Skills Required Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Skills Required</h2>';
  page += '<div class="card-container two-column">';

  // Get skills content from sections
  const skillsContent = sections['Skills Required']?.content || '';

  // Create a temporary div to parse the skills content
  const skillsDiv = document.createElement('div');
  skillsDiv.innerHTML = skillsContent;

  // Try to find technical and soft skills subsections
  const technicalHeading = findHeadingByText(skillsDiv, 'Technical Skills');
  const softHeading = findHeadingByText(skillsDiv, 'Soft Skills');

  if (technicalHeading && softHeading) {
    // Extract content for each subsection
    const technicalContent = extractContentAfterHeading(technicalHeading, 'H3');
    const softContent = extractContentAfterHeading(softHeading, 'H3');

    page += `<div class="info-card accent-blue">
              <h3>Technical Skills</h3>
              ${technicalContent || '<p>No specific technical skills information available.</p>'}
            </div>`;

    page += `<div class="info-card accent-green">
              <h3>Soft Skills</h3>
              ${softContent || '<p>No specific soft skills information available.</p>'}
            </div>`;
  } else {
    // If we couldn't find subsections, just show all skills
    page += `<div class="info-card full-width">
              ${skillsContent || '<p>No skills information available.</p>'}
            </div>`;
  }

  page += '</div></div>'; // End of Skills Required Section

  page += '</div>'; // End of page
  return page;
}

// Create the Career Progression page
function createProgressionPage(sections) {
  let page = '<div class="career-page">';

  // Career Progression Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Career Progression</h2>';
  page += '<div class="card-container multi-column">';

  // Get career progression content from sections
  const progressionContent = sections['Career Progression']?.content || '';

  if (progressionContent) {
    // Create a temporary div to parse the progression content
    const progressionDiv = document.createElement('div');
    progressionDiv.innerHTML = progressionContent;

    // Try to find subsections for different levels
    const entryLevelHeading = findHeadingByText(progressionDiv, 'Entry Level');
    const midLevelHeading = findHeadingByText(progressionDiv, 'Mid-Level');
    const seniorLevelHeading = findHeadingByText(progressionDiv, 'Senior Level');

    // Array to store all level headings we find
    const levelHeadings = [];
    if (entryLevelHeading) levelHeadings.push({ title: 'Entry Level', heading: entryLevelHeading, color: 'accent-blue' });
    if (midLevelHeading) levelHeadings.push({ title: 'Mid-Level', heading: midLevelHeading, color: 'accent-green' });
    if (seniorLevelHeading) levelHeadings.push({ title: 'Senior Level', heading: seniorLevelHeading, color: 'accent-orange' });

    // If we couldn't find the standard headings, try to find any H3 headings
    if (levelHeadings.length === 0) {
      const h3Headings = progressionDiv.querySelectorAll('h3');
      const colors = ['accent-blue', 'accent-green', 'accent-orange', 'accent-purple', 'accent-red'];

      h3Headings.forEach((heading, index) => {
        levelHeadings.push({
          title: heading.textContent,
          heading: heading,
          color: colors[index % colors.length]
        });
      });
    }

    if (levelHeadings.length > 0) {
      // Create a card for each career level
      levelHeadings.forEach((level) => {
        const content = extractContentAfterHeading(level.heading, 'H3');

        page += `<div class="info-card ${level.color}">
                  <h3>${level.title}</h3>
                  ${content}
                </div>`;
      });
    } else {
      // If no subsections, just show all progression content
      page += `<div class="info-card full-width">
                ${progressionContent || '<p>No detailed career progression information available.</p>'}
              </div>`;
    }
  } else {
    page += `<div class="info-card full-width">
              <p>No career progression information available.</p>
            </div>`;
  }

  page += '</div></div>'; // End of Career Progression Section

  page += '</div>'; // End of page
  return page;
}

// Create the Organizations & Certifications page
function createOrganizationsPage(sections) {
  let page = '<div class="career-page">';

  // Key Organizations Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Key Organizations & Institutions</h2>';
  page += '<div class="card-container">';

  // Get organizations content from sections
  const orgsContent = sections['Key Organizations & Institutions']?.content || '';

  page += `<div class="info-card accent-purple">${orgsContent || '<p>No key organizations information available.</p>'}</div>`;
  page += '</div></div>'; // End of Organizations Section

  // Certifications Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Certifications & Licensing</h2>';
  page += '<div class="card-container">';

  // Get certifications content from sections
  const certContent = sections['Certification & Licensing']?.content || '';

  page += `<div class="info-card accent-blue">${certContent || '<p>No certification or licensing information available.</p>'}</div>`;
  page += '</div></div>'; // End of Certifications Section

  // Industry Outlook Section (if available)
  if (sections['Industry Outlook']) {
    page += '<div class="section">';
    page += '<h2 class="section-title">Industry Outlook</h2>';
    page += '<div class="card-container">';

    const outlookContent = sections['Industry Outlook'].content;

    page += `<div class="info-card accent-green">${outlookContent}</div>`;
    page += '</div></div>'; // End of Industry Outlook Section
  }

  page += '</div>'; // End of page
  return page;
}

// Create the Tips & Resources page
function createTipsResourcesPage(sections) {
  let page = '<div class="career-page">';

  // Tips for Success Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Tips for Success</h2>';
  page += '<div class="card-container">';

  // Get tips content from sections
  const tipsContent = sections['Tips for Success']?.content || '';

  page += `<div class="info-card accent-orange">${tipsContent || '<p>No specific tips for success information available.</p>'}</div>`;
  page += '</div></div>'; // End of Tips Section

  // Resources Section
  page += '<div class="section">';
  page += '<h2 class="section-title">Resources for Further Learning</h2>';
  page += '<div class="card-container">';

  // Get resources content from sections
  const resourcesContent = sections['Resources for Further Learning']?.content || '';

  page += `<div class="info-card accent-blue">${resourcesContent || '<p>No specific resources for further learning available.</p>'}</div>`;
  page += '</div></div>'; // End of Resources Section

  // Challenges Section (if available)
  if (sections['Challenges']) {
    page += '<div class="section">';
    page += '<h2 class="section-title">Challenges & How to Overcome Them</h2>';
    page += '<div class="card-container">';

    const challengesContent = sections['Challenges'].content;

    page += `<div class="info-card accent-red">${challengesContent}</div>`;
    page += '</div></div>'; // End of Challenges Section
  }

  page += '</div>'; // End of page
  return page;
}

// Helper function to find a heading by text content
function findHeadingByText(container, text) {
  const headings = container.querySelectorAll('h2, h3');
  for (const heading of headings) {
    if (heading.textContent.toLowerCase().includes(text.toLowerCase())) {
      return heading;
    }
  }
  return null;
}

// Helper function to extract content after a heading until the next heading of same or higher level
function extractContentAfterHeading(heading, stopAtTag = 'H2') {
  if (!heading) return '';

  let content = '';
  let currentElement = heading.nextElementSibling;

  while (currentElement && currentElement.tagName !== stopAtTag) {
    content += currentElement.outerHTML;
    currentElement = currentElement.nextElementSibling;
  }

  return content;
}

// Helper function to extract content by keywords
function extractContentByKeywords(container, keywords) {
  const paragraphs = container.querySelectorAll('p');
  const lists = container.querySelectorAll('ul');
  let content = '';

  // Check paragraphs
  for (const p of paragraphs) {
    for (const keyword of keywords) {
      if (p.textContent.toLowerCase().includes(keyword.toLowerCase())) {
        content += p.outerHTML;

        // If followed by a list, include it
        let nextElement = p.nextElementSibling;
        if (nextElement && nextElement.tagName === 'UL') {
          content += nextElement.outerHTML;
        }
        break;
      }
    }
  }

  // Check list items
  for (const list of lists) {
    const items = list.querySelectorAll('li');
    for (const item of items) {
      for (const keyword of keywords) {
        if (item.textContent.toLowerCase().includes(keyword.toLowerCase())) {
          // Include the whole list if any item matches
          content += list.outerHTML;
          break;
        }
      }
    }
  }

  return content;
}

// Helper function to extract subheadings after a main heading
function extractSubheadings(mainHeading) {
  if (!mainHeading) return [];

  const subheadings = [];
  let currentElement = mainHeading.nextElementSibling;

  while (currentElement) {
    if (currentElement.tagName === 'H2') {
      break; // Stop at the next main heading
    }

    if (currentElement.tagName === 'H3') {
      subheadings.push({
        title: currentElement.textContent,
        heading: currentElement
      });
    }

    currentElement = currentElement.nextElementSibling;
  }

  return subheadings;
}

// Watch for changes in the content prop
watch(() => props.content, (newContent) => {
  processedPages.value = processContent(newContent);
  currentPage.value = 1; // Reset to first page when content changes
}, { immediate: true });

// Process content on mount
onMounted(() => {
  processedPages.value = processContent(props.content);
});
</script>

<style>
@import '~/assets/css/career-info-cards.css';

.paged-career-info {
  width: 100%;
}

.career-page {
  margin-bottom: 2rem;
}

.section {
  margin-bottom: 2.5rem;
}

.section-title {
  text-align: center;
  font-size: 1.5rem;
  font-weight: 600;
  color: white;
  background-color: #4A6163;
  margin-bottom: 1.5rem;
  padding: 0.75rem 1.25rem;
  border-radius: 0.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  position: relative;
}

/* Different colors for section titles */
.section:nth-child(2) .section-title {
  background-color: #5D8A66; /* Green */
}

.section:nth-child(3) .section-title {
  background-color: #D95D39; /* Orange */
}

.section:nth-child(4) .section-title {
  background-color: #7B506F; /* Purple */
}

.card-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.card-container.two-column,
.card-container.multi-column {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.info-card {
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  border-top: 4px solid #4A6163;
}

.info-card.full-width {
  grid-column: 1 / -1;
}

.info-card.accent-blue {
  border-top-color: #1A759F;
}

.info-card.accent-green {
  border-top-color: #5D8A66;
}

.info-card.accent-orange {
  border-top-color: #D95D39;
}

.info-card.accent-purple {
  border-top-color: #7B506F;
}

.info-card.accent-red {
  border-top-color: #D62828;
}

.info-card h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2F4858;
  margin-top: 0;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #F9C846;
}

.info-card ul {
  margin-bottom: 1rem;
  padding-left: 1.5rem;
}

.info-card ul li {
  position: relative;
  padding-left: 0.5rem;
  margin-bottom: 0.5rem;
}

.info-card p {
  margin-bottom: 1rem;
}

.info-card p:last-child {
  margin-bottom: 0;
}

@media (max-width: 768px) {
  .card-container.two-column,
  .card-container.multi-column {
    grid-template-columns: 1fr;
  }

  .section-title {
    font-size: 1.3rem;
  }

  .info-card {
    padding: 1.25rem;
  }
}
</style>
