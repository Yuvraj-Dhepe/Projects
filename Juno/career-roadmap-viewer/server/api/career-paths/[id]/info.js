import { getCareerPathInfoAsHtml } from '../../../utils/markdownConverter';

// Check if we're in debug mode
const config = useRuntimeConfig();
const DEBUG = config.debugMode === 'true';

/**
 * Debug logger that only logs when DEBUG is true
 * @param {string} message - The message to log
 * @param {...any} args - Additional arguments to log
 */
function debugLog(message, ...args) {
  if (DEBUG) {
    console.log(`[CAREER PATH INFO API] ${message}`, ...args);
  }
}

export default defineEventHandler(async (event) => {
  try {
    const id = event.context.params.id;
    debugLog(`Requested career path info for: ${id}`);

    if (!id) {
      debugLog('No ID provided');
      return {
        statusCode: 400,
        body: 'Career path ID is required'
      };
    }

    // Get the career path info from the markdown file
    debugLog('Attempting to get career path info from markdown file');
    const careerPathInfo = getCareerPathInfoAsHtml(id);

    if (careerPathInfo) {
      debugLog('Successfully retrieved career path info from markdown file');
      return {
        html: careerPathInfo
      };
    }

    // If we couldn't get the info from the markdown file, try the static API
    debugLog('No info found in markdown file, trying static API');

    // Make a request to the static-career-paths API
    const staticApiUrl = `/api/static-career-paths/${id}?format=html`;
    debugLog(`Requesting from static API: ${staticApiUrl}`);

    try {
      // Use $fetch to make an internal API call
      const staticResponse = await $fetch(staticApiUrl);

      if (staticResponse && staticResponse.html) {
        debugLog('Successfully retrieved info from static API');
        return {
          html: staticResponse.html
        };
      }
    } catch (staticError) {
      debugLog('Error fetching from static API:', staticError.message);
    }

    // If we still don't have any info, return a 404
    debugLog('No career path info found in any source');
    return {
      statusCode: 404,
      body: `Career path information not found for ID: ${id}`
    };
  } catch (error) {
    console.error('Error fetching career path info:', error);
    debugLog('Error fetching career path info:', error.message, error.stack);
    return {
      statusCode: 500,
      body: 'Error fetching career path information'
    };
  }
});
