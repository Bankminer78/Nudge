import Foundation

// Represents a single clickable item, like a specific concert or apartment listing.
struct UpdateItem: Identifiable {
    let id = UUID()
    let title: String
    let link: URL
}

// Represents a category of updates, like "Concerts" or "Apartments".
// This is the main object our view's state will hold.
struct UpdateCategory: Identifiable {
    let id = UUID()
    let title: String
    let items: [UpdateItem]
}

class MockData {
    private static var checkCount = 0
    
    // The completion handler now returns an array of UpdateCategory
    static func checkForUpdates(completion: @escaping ([UpdateCategory]?) -> Void) {
        checkCount += 1
        
        // Simulate finding updates every 10 seconds
        if checkCount >= 10 {
            checkCount = 0
            
            // 🎵 Sample Concert Data
            let concertCategory = UpdateCategory(
                title: "Concerts",
                items: [
                    UpdateItem(title: "The National @ Madison Square Garden", link: URL(string: "https://www.ticketmaster.com")!),
                    UpdateItem(title: "LCD Soundsystem @ Knockdown Center", link: URL(string: "https://www.axs.com")!),
                    UpdateItem(title: "Vampire Weekend @ Barclays Center", link: URL(string: "https://www.ticketmaster.com")!)
                ]
            )
            
            // 🏢 Sample Apartment Data
            let apartmentCategory = UpdateCategory(
                title: "Apartments",
                items: [
                    UpdateItem(title: "2 Bed, 1 Bath in Williamsburg", link: URL(string: "https://www.zillow.com")!),
                    UpdateItem(title: "Studio Loft in SoHo with City Views", link: URL(string: "https://streeteasy.com")!),
                    UpdateItem(title: "1 Bed with Balcony in Park Slope", link: URL(string: "https://www.zillow.com")!)
                ]
            )
            
            // Return both categories
            let newUpdates = [concertCategory, apartmentCategory]
            completion(newUpdates)
            
        } else {
            // Return nil when no new updates are found
            completion(nil)
        }
    }
}
