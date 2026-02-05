"""
Comprehensive XML to CSV Converter for Bus Data
Extracts ALL available fields from TransXChange XML files
Handles multiple data levels: stop-to-stop, route-level, schedules, etc.
"""

import xml.etree.ElementTree as ET
import csv
import os
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


class XMLtoCSVExtractor:
    def __init__(self):
        self.namespace = {'txc': 'http://www.transxchange.org.uk/'}
        self.all_records = []
        self.field_stats = defaultdict(int)  # Track which fields we found
        self.total_files_processed = 0
        
    def parse_duration(self, duration_str):
        """Convert ISO 8601 duration (PT1M, PT50S) to seconds"""
        if not duration_str:
            return None
        
        # Remove 'PT' prefix
        duration_str = duration_str.replace('PT', '')
        
        total_seconds = 0
        # Extract hours, minutes, seconds
        hours = re.findall(r'(\d+)H', duration_str)
        minutes = re.findall(r'(\d+)M', duration_str)
        seconds = re.findall(r'(\d+\.?\d*)S', duration_str)
        
        if hours:
            total_seconds += int(hours[0]) * 3600
        if minutes:
            total_seconds += int(minutes[0]) * 60
        if seconds:
            total_seconds += float(seconds[0])
            
        return total_seconds if total_seconds > 0 else None
    
    def extract_text(self, element, path, namespaced=True):
        """Safely extract text from XML element"""
        if element is None:
            return None
        
        if namespaced:
            found = element.find(path, self.namespace)
        else:
            found = element.find(path)
            
        return found.text if found is not None else None
    
    def extract_attribute(self, element, attr_name):
        """Safely extract attribute from XML element"""
        if element is None:
            return None
        return element.get(attr_name)
    
    def build_stop_lookup(self, root):
        """Build a lookup dictionary for stop information"""
        stop_lookup = {}
        
        stop_points = root.find('txc:StopPoints', self.namespace)
        if stop_points is not None:
            for stop in stop_points.findall('txc:AnnotatedStopPointRef', self.namespace):
                stop_id = self.extract_text(stop, 'txc:StopPointRef')
                if stop_id:
                    stop_lookup[stop_id] = {
                        'CommonName': self.extract_text(stop, 'txc:CommonName'),
                        'Indicator': self.extract_text(stop, 'txc:Indicator'),
                        'LocalityName': self.extract_text(stop, 'txc:LocalityName'),
                        'LocalityQualifier': self.extract_text(stop, 'txc:LocalityQualifier'),
                        'Latitude': self.extract_text(stop, 'txc:Location/txc:Latitude'),
                        'Longitude': self.extract_text(stop, 'txc:Location/txc:Longitude'),
                    }
        
        return stop_lookup
    
    def build_route_link_lookup(self, root):
        """Build a lookup dictionary for route links (with distance)"""
        route_link_lookup = {}
        
        route_sections = root.find('txc:RouteSections', self.namespace)
        if route_sections is not None:
            for route_section in route_sections.findall('txc:RouteSection', self.namespace):
                for route_link in route_section.findall('txc:RouteLink', self.namespace):
                    link_id = self.extract_attribute(route_link, 'id')
                    
                    from_elem = route_link.find('txc:From', self.namespace)
                    to_elem = route_link.find('txc:To', self.namespace)
                    
                    from_stop = self.extract_text(from_elem, 'txc:StopPointRef') if from_elem is not None else None
                    to_stop = self.extract_text(to_elem, 'txc:StopPointRef') if to_elem is not None else None
                    distance = self.extract_text(route_link, 'txc:Distance')
                    
                    if link_id:
                        route_link_lookup[link_id] = {
                            'FromStop': from_stop,
                            'ToStop': to_stop,
                            'Distance': distance
                        }
        
        return route_link_lookup
    
    def extract_journey_pattern_sections(self, root, stop_lookup, route_link_lookup, filename):
        """Extract stop-to-stop journey data with timing and distance"""
        records = []
        
        # Get metadata from root
        creation_date = self.extract_attribute(root, 'CreationDateTime')
        modification_date = self.extract_attribute(root, 'ModificationDateTime')
        
        # Get operator info
        operator_code = None
        operator_name = None
        operators = root.find('txc:Operators', self.namespace)
        if operators is not None:
            operator = operators.find('txc:Operator', self.namespace)
            if operator is not None:
                operator_code = self.extract_text(operator, 'txc:OperatorCode')
                operator_name = self.extract_text(operator, 'txc:OperatorNameOnLicence')
        
        # Get service info
        services = root.find('txc:Services', self.namespace)
        if services is not None:
            for service in services.findall('txc:Service', self.namespace):
                service_code = self.extract_text(service, 'txc:ServiceCode')
                
                # Get line info
                lines = service.findall('txc:Lines/txc:Line', self.namespace)
                line_name = None
                if lines:
                    line_name = self.extract_text(lines[0], 'txc:LineName')
                
                # Get operating period
                operating_period = service.find('txc:OperatingPeriod', self.namespace)
                start_date = None
                end_date = None
                if operating_period is not None:
                    start_date = self.extract_text(operating_period, 'txc:StartDate')
                    end_date = self.extract_text(operating_period, 'txc:EndDate')
                
                # Process journey patterns
                journey_patterns = service.findall('.//txc:JourneyPattern', self.namespace)
                for jp in journey_patterns:
                    direction = self.extract_text(jp, 'txc:Direction')
                    route_ref = self.extract_text(jp, 'txc:RouteRef')
                    jp_section_refs = jp.findall('txc:JourneyPatternSectionRefs', self.namespace)
                    
                    # Get destination display
                    destination = self.extract_text(jp, 'txc:DestinationDisplay')
        
        # Extract journey pattern timing links
        jp_sections = root.find('txc:JourneyPatternSections', self.namespace)
        if jp_sections is not None:
            for jp_section in jp_sections.findall('txc:JourneyPatternSection', self.namespace):
                jp_section_id = self.extract_attribute(jp_section, 'id')
                
                timing_links = jp_section.findall('txc:JourneyPatternTimingLink', self.namespace)
                for timing_link in timing_links:
                    timing_link_id = self.extract_attribute(timing_link, 'id')
                    
                    # Extract FROM information
                    from_elem = timing_link.find('txc:From', self.namespace)
                    from_stop_id = None
                    from_sequence = None
                    from_activity = None
                    from_timing_status = None
                    from_wait_time = None
                    
                    if from_elem is not None:
                        from_stop_id = self.extract_text(from_elem, 'txc:StopPointRef')
                        from_sequence = self.extract_attribute(from_elem, 'SequenceNumber')
                        from_activity = self.extract_text(from_elem, 'txc:Activity')
                        from_timing_status = self.extract_text(from_elem, 'txc:TimingStatus')
                        wait_time_raw = self.extract_text(from_elem, 'txc:WaitTime')
                        from_wait_time = self.parse_duration(wait_time_raw) if wait_time_raw else None
                    
                    # Extract TO information
                    to_elem = timing_link.find('txc:To', self.namespace)
                    to_stop_id = None
                    to_sequence = None
                    to_activity = None
                    to_timing_status = None
                    to_wait_time = None
                    
                    if to_elem is not None:
                        to_stop_id = self.extract_text(to_elem, 'txc:StopPointRef')
                        to_sequence = self.extract_attribute(to_elem, 'SequenceNumber')
                        to_activity = self.extract_text(to_elem, 'txc:Activity')
                        to_timing_status = self.extract_text(to_elem, 'txc:TimingStatus')
                        wait_time_raw = self.extract_text(to_elem, 'txc:WaitTime')
                        to_wait_time = self.parse_duration(wait_time_raw) if wait_time_raw else None
                    
                    # Extract runtime
                    runtime_raw = self.extract_text(timing_link, 'txc:RunTime')
                    runtime_seconds = self.parse_duration(runtime_raw) if runtime_raw else None
                    
                    # Get route link reference
                    route_link_ref = self.extract_text(timing_link, 'txc:RouteLinkRef')
                    
                    # Get distance from route link lookup
                    distance = None
                    if route_link_ref and route_link_ref in route_link_lookup:
                        distance = route_link_lookup[route_link_ref].get('Distance')
                    
                    # Get stop details from lookup
                    from_stop_info = stop_lookup.get(from_stop_id, {})
                    to_stop_info = stop_lookup.get(to_stop_id, {})
                    
                    # Build comprehensive record
                    record = {
                        # File metadata
                        'source_file': filename,
                        'creation_date': creation_date,
                        'modification_date': modification_date,
                        
                        # Operator info
                        'operator_code': operator_code,
                        'operator_name': operator_name,
                        
                        # Service info
                        'service_code': service_code if 'service_code' in locals() else None,
                        'line_name': line_name if 'line_name' in locals() else None,
                        'operating_start_date': start_date if 'start_date' in locals() else None,
                        'operating_end_date': end_date if 'end_date' in locals() else None,
                        
                        # Journey pattern info
                        'journey_pattern_section_id': jp_section_id,
                        'timing_link_id': timing_link_id,
                        'route_link_ref': route_link_ref,
                        
                        # FROM stop information
                        'from_stop_id': from_stop_id,
                        'from_stop_name': from_stop_info.get('CommonName'),
                        'from_stop_indicator': from_stop_info.get('Indicator'),
                        'from_locality_name': from_stop_info.get('LocalityName'),
                        'from_locality_qualifier': from_stop_info.get('LocalityQualifier'),
                        'from_latitude': from_stop_info.get('Latitude'),
                        'from_longitude': from_stop_info.get('Longitude'),
                        'from_sequence_number': from_sequence,
                        'from_activity': from_activity,
                        'from_timing_status': from_timing_status,
                        'from_wait_time_seconds': from_wait_time,
                        
                        # TO stop information
                        'to_stop_id': to_stop_id,
                        'to_stop_name': to_stop_info.get('CommonName'),
                        'to_stop_indicator': to_stop_info.get('Indicator'),
                        'to_locality_name': to_stop_info.get('LocalityName'),
                        'to_locality_qualifier': to_stop_info.get('LocalityQualifier'),
                        'to_latitude': to_stop_info.get('Latitude'),
                        'to_longitude': to_stop_info.get('Longitude'),
                        'to_sequence_number': to_sequence,
                        'to_activity': to_activity,
                        'to_timing_status': to_timing_status,
                        'to_wait_time_seconds': to_wait_time,
                        
                        # Travel metrics (KEY FOR ML)
                        'runtime_seconds': runtime_seconds,
                        'runtime_minutes': round(runtime_seconds / 60, 2) if runtime_seconds else None,
                        'distance_meters': distance,
                        'distance_km': round(float(distance) / 1000, 3) if distance else None,
                    }
                    
                    # Track field statistics
                    for key, value in record.items():
                        if value is not None and value != '':
                            self.field_stats[key] += 1
                    
                    records.append(record)
        
        return records
    
    def process_xml_file(self, xml_path):
        """Process a single XML file and extract all data"""
        print(f"\n📄 Processing: {os.path.basename(xml_path)}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Build lookups
            stop_lookup = self.build_stop_lookup(root)
            route_link_lookup = self.build_route_link_lookup(root)
            
            print(f"   Found {len(stop_lookup)} stops, {len(route_link_lookup)} route links")
            
            # Extract journey pattern sections (main data)
            filename = os.path.basename(xml_path)
            records = self.extract_journey_pattern_sections(
                root, stop_lookup, route_link_lookup, filename
            )
            
            print(f"   ✓ Extracted {len(records)} stop-to-stop records")
            
            self.all_records.extend(records)
            self.total_files_processed += 1
            
        except Exception as e:
            print(f"   ✗ Error processing file: {e}")
    
    def process_directory(self, directory_path):
        """Process all XML files in a directory"""
        xml_files = list(Path(directory_path).glob('*.xml'))
        
        print(f"\n{'='*70}")
        print(f"🚀 COMPREHENSIVE XML TO CSV EXTRACTOR")
        print(f"{'='*70}")
        print(f"\n📁 Directory: {directory_path}")
        print(f"📊 Found {len(xml_files)} XML files")
        
        for xml_file in xml_files:
            self.process_xml_file(xml_file)
        
        return self.all_records
    
    def save_to_csv(self, output_path):
        """Save all extracted records to CSV"""
        if not self.all_records:
            print("\n⚠️  No records to save!")
            return
        
        # Get all possible field names
        all_fields = set()
        for record in self.all_records:
            all_fields.update(record.keys())
        
        # Sort fields logically
        field_order = [
            'source_file', 'creation_date', 'modification_date',
            'operator_code', 'operator_name', 'service_code', 'line_name',
            'operating_start_date', 'operating_end_date',
            'journey_pattern_section_id', 'timing_link_id', 'route_link_ref',
            'from_stop_id', 'from_stop_name', 'from_stop_indicator',
            'from_locality_name', 'from_locality_qualifier',
            'from_latitude', 'from_longitude',
            'from_sequence_number', 'from_activity', 'from_timing_status', 'from_wait_time_seconds',
            'to_stop_id', 'to_stop_name', 'to_stop_indicator',
            'to_locality_name', 'to_locality_qualifier',
            'to_latitude', 'to_longitude',
            'to_sequence_number', 'to_activity', 'to_timing_status', 'to_wait_time_seconds',
            'runtime_seconds', 'runtime_minutes', 'distance_meters', 'distance_km'
        ]
        
        # Add any fields not in the predefined order
        remaining_fields = sorted(all_fields - set(field_order))
        final_field_order = field_order + remaining_fields
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_field_order)
            writer.writeheader()
            writer.writerows(self.all_records)
        
        print(f"\n✅ CSV saved: {output_path}")
        print(f"   Total records: {len(self.all_records)}")
    
    def print_summary(self):
        """Print detailed summary of extraction"""
        print(f"\n{'='*70}")
        print(f"📊 EXTRACTION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📁 Files processed: {self.total_files_processed}")
        print(f"📝 Total records extracted: {len(self.all_records)}")
        
        if not self.all_records:
            print("\n⚠️  No data extracted!")
            return
        
        # Calculate data completeness
        print(f"\n{'='*70}")
        print(f"🎯 DATA COMPLETENESS (by field)")
        print(f"{'='*70}")
        
        total_records = len(self.all_records)
        
        # Group fields by category
        categories = {
            'HIGH PRIORITY (ML-Ready Fields)': [
                'from_stop_id', 'to_stop_id', 'from_stop_name', 'to_stop_name',
                'runtime_seconds', 'distance_meters', 'from_latitude', 'from_longitude',
                'to_latitude', 'to_longitude'
            ],
            'MEDIUM PRIORITY (Route/Service Info)': [
                'line_name', 'service_code', 'operator_name', 'from_sequence_number',
                'to_sequence_number', 'route_link_ref'
            ],
            'LOW PRIORITY (Metadata)': [
                'source_file', 'creation_date', 'from_activity', 'to_activity',
                'from_timing_status', 'to_timing_status'
            ]
        }
        
        for category, fields in categories.items():
            print(f"\n{category}:")
            print("-" * 70)
            for field in fields:
                if field in self.field_stats:
                    count = self.field_stats[field]
                    percentage = (count / total_records) * 100
                    status = "✓" if percentage > 80 else "⚠" if percentage > 50 else "✗"
                    print(f"  {status} {field:30s}: {count:6d} / {total_records:6d} ({percentage:5.1f}%)")
                else:
                    print(f"  ✗ {field:30s}:      0 / {total_records:6d} (  0.0%)")
        
        # Calculate overall ML readiness
        print(f"\n{'='*70}")
        print(f"🤖 ML READINESS SCORE")
        print(f"{'='*70}")
        
        ml_critical_fields = [
            'from_stop_id', 'to_stop_id', 'runtime_seconds', 'distance_meters'
        ]
        
        ml_scores = []
        for field in ml_critical_fields:
            if field in self.field_stats:
                score = (self.field_stats[field] / total_records) * 100
                ml_scores.append(score)
        
        if ml_scores:
            avg_ml_score = sum(ml_scores) / len(ml_scores)
            print(f"\n  Average completeness of critical ML fields: {avg_ml_score:.1f}%")
            
            if avg_ml_score >= 90:
                print("  ✓✓✓ EXCELLENT - Data is highly suitable for ML training!")
            elif avg_ml_score >= 70:
                print("  ✓✓  GOOD - Data is suitable for ML with some limitations")
            elif avg_ml_score >= 50:
                print("  ✓   MODERATE - Data can be used but may need preprocessing")
            else:
                print("  ✗   LIMITED - Data may not be sufficient for robust ML models")
        
        # Sample records
        print(f"\n{'='*70}")
        print(f"📋 SAMPLE RECORDS (first 3)")
        print(f"{'='*70}")
        
        for i, record in enumerate(self.all_records[:3], 1):
            print(f"\nRecord {i}:")
            print(f"  Route: {record.get('from_stop_name', 'N/A')} → {record.get('to_stop_name', 'N/A')}")
            print(f"  Runtime: {record.get('runtime_seconds', 'N/A')} seconds ({record.get('runtime_minutes', 'N/A')} min)")
            print(f"  Distance: {record.get('distance_meters', 'N/A')} meters ({record.get('distance_km', 'N/A')} km)")
            print(f"  Line: {record.get('line_name', 'N/A')}")
            print(f"  Operator: {record.get('operator_name', 'N/A')}")


def main():
    """Main execution function"""
    
    # Get the directory containing the XML files
    script_dir = Path(__file__).parent
    xml_directory = script_dir / "Metroline_TfL_9w6A35W"
    
    # Output CSV file
    output_csv = script_dir / "bus_data_comprehensive.csv"
    
    # Create extractor and process files
    extractor = XMLtoCSVExtractor()
    
    # Process all XML files
    extractor.process_directory(xml_directory)
    
    # Save to CSV
    extractor.save_to_csv(output_csv)
    
    # Print detailed summary
    extractor.print_summary()
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📄 Output file: {output_csv}")
    print(f"\n💡 Next steps:")
    print(f"   1. Open the CSV in Excel/Python to explore the data")
    print(f"   2. Check field completeness for your specific use case")
    print(f"   3. For ML: Focus on records with runtime and distance data")
    print(f"   4. Consider data cleaning/preprocessing as needed")
    print()


if __name__ == "__main__":
    main()
