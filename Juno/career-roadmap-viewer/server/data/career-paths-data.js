/**
 * Hardcoded career paths data to ensure availability in serverless environments
 * This file serves as a fallback when static data files cannot be accessed
 */

export const hardcodedCareerPaths = [
  {
    id: "air_hostess",
    name: "Air Hostess"
  },
  {
    id: "armed_forces",
    name: "Armed Forces"
  },
  {
    id: "bank_po",
    name: "Bank PO"
  },
  {
    id: "ca",
    name: "CA"
  },
  {
    id: "civil_services",
    name: "Civil Services"
  },
  {
    id: "cs",
    name: "CS"
  },
  {
    id: "fashion_designing",
    name: "Fashion Designing"
  },
  {
    id: "hotel_management",
    name: "Hotel Management"
  },
  {
    id: "law",
    name: "Law"
  },
  {
    id: "mba",
    name: "MBA"
  },
  {
    id: "medical",
    name: "Medical"
  },
  {
    id: "merchant_navy",
    name: "Merchant Navy"
  }
];

/**
 * Get career paths data
 * @returns {Array} Array of career path objects
 */
export function getCareerPathsData() {
  return hardcodedCareerPaths;
}

/**
 * Get a specific career path by ID
 * @param {string} id - The career path ID
 * @returns {Object|null} The career path object or null if not found
 */
export function getCareerPathById(id) {
  return hardcodedCareerPaths.find(path => path.id === id) || null;
}
